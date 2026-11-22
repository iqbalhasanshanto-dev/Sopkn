"""
Spokn - Personal Offline Dictation for Apple Silicon

Architecture & V1 Flow:
1. UI Initialization & Model Warm-Up:
   - Initialize PySide6 QApplication and display the DictationPillOverlay in IDLE state.
   - Run MLX Whisper warm-up asynchronously in a background thread so the UI appears instantly.
2. Press-to-Toggle Dictation (Right Option or Click-to-Toggle):
   - Press Right Option (HOTKEY) or click the IDLE pill to start microphone recording (LISTENING).
   - Audio thread computes RMS volume internally without emitting Qt signals, avoiding deadlocks.
   - Qt main thread polls latest RMS on a 60 ms timer: transitions to SPEAKING once RMS >= SPEECH_THRESHOLD,
     modulating the live waveform smoothly.
3. Stop or Cancel:
   - Press Right Option again (or click Stop on the pill) to stop recording and transition to COLLAPSING -> PROCESSING.
   - Click × Cancel at any time to discard audio and immediately return to IDLE.
4. Local Inference, Corrections & Paste:
   - In-memory float32 audio is transcribed with mlx-whisper on a background thread.
   - Word/phrase corrections from corrections.json are applied automatically.
   - Copies transcript to clipboard, simulates Cmd+V via osascript, and restores prior clipboard.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time
import numpy as np
from pynput import keyboard
from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtWidgets import QApplication
import sounddevice as sd
import mlx_whisper

import config
from overlay import DictationPillOverlay, PillState


class UiBridge(QObject):
    """Thread-safe signal bridge communicating between background threads and the Qt UI."""

    hotkey_toggled = Signal()
    state_changed = Signal(object)  # PillState


def apply_corrections(text: str) -> str:
    """Apply word/phrase replacements from corrections.json using whole-word regex."""
    if not text or not isinstance(text, str):
        return ""

    corrections_path = Path(__file__).parent / "corrections.json"
    if not corrections_path.exists():
        return text

    try:
        with open(corrections_path, "r", encoding="utf-8") as f:
            replacements: dict[str, str] = json.load(f)
    except Exception as exc:
        print(f"[{config.APP_NAME}] Warning reading corrections.json: {exc}", flush=True)
        return text

    corrected = text
    for pattern, replacement in replacements.items():
        # Match whole word/phrase with case-insensitivity
        regex = rf"\b{re.escape(pattern)}\b"
        corrected = re.sub(regex, replacement, corrected, flags=re.IGNORECASE)

    return corrected


class AudioRecorder:
    """Handles real-time microphone audio capture and normalization using sounddevice.

    Thread Safety:
    - Never calls Qt functions or emits Qt signals from _audio_callback (the PortAudio
      real-time audio thread).
    - Computes RMS volume and stores it in self._latest_rms for the Qt main thread
      to poll safely at its own frequency via QTimer.
    """

    def __init__(self) -> None:
        """Initializes internal audio chunk buffer, recording flag, and stream handle."""
        self._buffer: list[np.ndarray] = []
        self._is_recording: bool = False
        self._stream: sd.InputStream | None = None
        self._latest_rms: float = 0.0

    @property
    def is_recording(self) -> bool:
        """Return whether audio is currently actively recording."""
        return self._is_recording

    @property
    def latest_rms(self) -> float:
        """Return the most recent RMS volume computed on the audio thread."""
        return self._latest_rms

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: dict,
        status: sd.CallbackFlags,
    ) -> None:
        """Callback invoked by PortAudio on a dedicated audio thread for each block."""
        if status:
            print(f"[{config.APP_NAME}] Audio stream warning: {status}", flush=True)

        if self._is_recording:
            # indata is a view managed by sounddevice and gets overwritten;
            # store an explicit copy of each chunk.
            self._buffer.append(indata.copy())

            # Compute normalized RMS amplitude (0.0 to 1.0) from int16 audio
            # Fast, lock-free float assignment (no Qt calls, no callbacks)
            data_float = indata.astype(np.float32) / 32768.0
            self._latest_rms = float(np.sqrt(np.mean(data_float ** 2)))

    def start(self) -> bool:
        """Reset buffer and open/start the sounddevice InputStream."""
        if self._is_recording:
            return True

        self._buffer.clear()
        self._latest_rms = 0.0
        try:
            self._stream = sd.InputStream(
                samplerate=config.SAMPLE_RATE,
                channels=config.CHANNELS,
                dtype=config.DTYPE,
                callback=self._audio_callback,
            )
            self._stream.start()
            self._is_recording = True
            return True
        except Exception as exc:
            print(
                f"[{config.APP_NAME}] Failed to start audio recording: {exc}\n"
                f"Please ensure microphone permissions are granted and PortAudio is installed.",
                flush=True,
            )
            self._is_recording = False
            if self._stream is not None:
                try:
                    self._stream.close()
                except Exception:
                    pass
                self._stream = None
            return False

    def stop(self) -> np.ndarray | None:
        """Stop and close the audio stream, then concatenate and normalize captured audio."""
        self._is_recording = False
        self._latest_rms = 0.0

        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as exc:
                print(f"[{config.APP_NAME}] Warning closing audio stream: {exc}", flush=True)
            finally:
                self._stream = None

        if not self._buffer:
            return None

        # Concatenate all captured chunks along axis 0 and flatten to 1D
        audio_int16 = np.concatenate(self._buffer, axis=0).flatten()

        duration = len(audio_int16) / config.SAMPLE_RATE
        if duration < config.MIN_RECORDING_DURATION:
            # Reject recordings that are too short to avoid false triggers
            return None

        # Normalize int16 PCM to float32 in range [-1.0, 1.0]
        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        return audio_float32

    def cancel(self) -> None:
        """Stop and close the audio stream, discarding captured chunks without returning audio."""
        self._is_recording = False
        self._latest_rms = 0.0
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as exc:
                print(f"[{config.APP_NAME}] Warning canceling audio stream: {exc}", flush=True)
            finally:
                self._stream = None
        self._buffer.clear()


class Transcriber:
    """Manages local speech-to-text inference with mlx-whisper on Apple Silicon."""

    def __init__(self, model_repo: str = config.MODEL_REPO) -> None:
        """Initialize Transcriber with the configured Hugging Face / MLX model repo."""
        self.model_repo = model_repo

    def warm_up(self) -> None:
        """Warm up the MLX model and Apple Silicon unified memory before live dictation."""
        print(f"[{config.APP_NAME}] Loading model ({self.model_repo})...", flush=True)
        try:
            # 1 second of silent float32 audio at 16kHz
            silent_audio = np.zeros(config.SAMPLE_RATE, dtype=np.float32)
            mlx_whisper.transcribe(
                silent_audio,
                path_or_hf_repo=self.model_repo,
                language=config.LANGUAGE,
                initial_prompt=config.INITIAL_PROMPT,
            )
            print(f"[{config.APP_NAME}] Model ready.", flush=True)
        except Exception as exc:
            print(
                f"[{config.APP_NAME}] Warning during model warm-up: {exc}",
                flush=True,
            )

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe a normalized 1D float32 numpy audio buffer to text with language & vocabulary hints."""
        try:
            result = mlx_whisper.transcribe(
                audio,
                path_or_hf_repo=self.model_repo,
                language=config.LANGUAGE,
                initial_prompt=config.INITIAL_PROMPT,
            )
            return result.get("text", "").strip()
        except Exception as exc:
            print(
                f"[{config.APP_NAME}] Transcription error: {exc}",
                flush=True,
            )
            return ""


def get_clipboard() -> str | None:
    """Read the current text content from the macOS system clipboard using pbpaste."""
    try:
        result = subprocess.run(
            ["pbpaste"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout
        return None
    except Exception:
        return None


def set_clipboard(text: str) -> None:
    """Write text to the macOS system clipboard using pbcopy."""
    subprocess.run(
        ["pbcopy"],
        input=text,
        text=True,
        check=True,
    )


def paste_text(text: str) -> bool:
    """Copy text to clipboard, simulate Cmd+V via osascript, and restore prior clipboard."""
    previous_clipboard = get_clipboard()
    try:
        set_clipboard(text)
        subprocess.run(
            [
                "osascript",
                "-e",
                'tell application "System Events" to keystroke "v" using command down',
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        time.sleep(config.CLIPBOARD_RESTORE_DELAY)
        return True
    except Exception as exc:
        print(
            f"[{config.APP_NAME}] Failed to paste text: {exc}\n"
            f"Please ensure Accessibility permissions are granted under "
            f"System Settings -> Privacy & Security -> Accessibility.",
            flush=True,
        )
        return False
    finally:
        if previous_clipboard is not None:
            try:
                set_clipboard(previous_clipboard)
            except Exception:
                pass


def get_hotkey() -> keyboard.Key:
    """Resolve the configured hotkey string to a pynput keyboard.Key enum value."""
    hotkey = getattr(keyboard.Key, config.HOTKEY, None)
    if hotkey is None:
        raise ValueError(
            f"Invalid hotkey '{config.HOTKEY}' specified in config.py. "
            f"Expected a valid attribute of pynput.keyboard.Key (e.g. 'alt_r', 'ctrl_r')."
        )
    return hotkey


class DictationApp:
    """Coordinates the Dictation Pill UI, global hotkey listener, and audio transcription."""

    def __init__(
        self,
        overlay: DictationPillOverlay | None = None,
        transcriber: Transcriber | None = None,
    ) -> None:
        self.overlay = overlay or DictationPillOverlay()
        self.transcriber = transcriber or Transcriber()
        self.target_hotkey = get_hotkey()

        # Thread-safe Qt signal bridge for cross-thread events
        self.ui_bridge = UiBridge()
        self.ui_bridge.hotkey_toggled.connect(self.handle_hotkey_toggle)
        self.ui_bridge.state_changed.connect(self.overlay.set_state)

        # Connect overlay interactive mouse events directly on Qt main thread
        self.overlay.idle_clicked.connect(self.start_dictation)
        self.overlay.cancel_clicked.connect(self.cancel_dictation)
        self.overlay.stop_clicked.connect(self.stop_dictation)

        # Audio recorder (no Qt callbacks inside audio thread)
        self.recorder = AudioRecorder()

        # 33 ms (~30 FPS) polling timer on Qt main thread for reading latest RMS volume
        self._level_timer = QTimer()
        self._level_timer.setInterval(33)
        self._level_timer.timeout.connect(self._poll_audio_level)

        # State tracking
        self.is_ready = False
        self.is_processing = False
        self._current_state = PillState.IDLE
        self._lock = threading.Lock()
        self._listener: keyboard.Listener | None = None
        self._last_hotkey_press_time: float = 0.0

    def start(self) -> None:
        """Show overlay, start global keyboard listener, and warm up model in background."""
        self.overlay.show()

        # Start pynput global keyboard listener in background thread
        self._listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release,
        )
        self._listener.start()

        # Run model warm-up in a background thread so the UI is immediately visible
        def _warmup_worker() -> None:
            self.transcriber.warm_up()
            with self._lock:
                self.is_ready = True
            print(
                f"[{config.APP_NAME}] Ready. Press Right Option or click pill to talk (Ctrl+C to quit).",
                flush=True,
            )

        warmup_thread = threading.Thread(
            target=_warmup_worker,
            daemon=True,
            name="ModelWarmupThread",
        )
        warmup_thread.start()

    # -------------------------------------------------------------------------
    # Dictation Control Logic (Always executed on Qt Main Thread)
    # -------------------------------------------------------------------------

    def handle_hotkey_toggle(self) -> None:
        """Toggle dictation on/off from the hotkey on the Qt main thread."""
        if self.recorder.is_recording:
            self.stop_dictation()
        else:
            self.start_dictation()

    def start_dictation(self) -> None:
        """Start audio recording, transition UI to LISTENING state, and start RMS polling."""
        with self._lock:
            if not self.is_ready or self.recorder.is_recording or self.is_processing:
                return

        if self.recorder.start():
            self._current_state = PillState.LISTENING
            self.overlay.set_state(PillState.LISTENING)
            self._level_timer.start()
            print(f"[{config.APP_NAME}] Listening...", flush=True)

    def stop_dictation(self) -> None:
        """Stop recording, trigger COLLAPSING -> PROCESSING, and transcribe."""
        if not self.recorder.is_recording:
            return

        self._level_timer.stop()
        self.overlay.set_audio_level(0.0)

        audio = self.recorder.stop()
        if audio is None:
            self._current_state = PillState.IDLE
            self.overlay.set_state(PillState.IDLE)
            print(f"[{config.APP_NAME}] Ignored short recording.", flush=True)
            print(f"[{config.APP_NAME}] Ready.", flush=True)
            return

        # Transition to COLLAPSING (which auto-morphs to PROCESSING after 220 ms)
        self._current_state = PillState.PROCESSING
        self.overlay.set_state(PillState.COLLAPSING)
        print(f"[{config.APP_NAME}] Transcribing...", flush=True)

        worker = threading.Thread(
            target=self.process_audio,
            args=(audio,),
            daemon=True,
            name="AudioProcessorThread",
        )
        worker.start()

    def cancel_dictation(self) -> None:
        """Cancel recording immediately and return to IDLE without transcribing."""
        self._level_timer.stop()
        self.overlay.set_audio_level(0.0)
        self.recorder.cancel()
        self._current_state = PillState.IDLE
        self.overlay.set_state(PillState.IDLE)
        print(f"[{config.APP_NAME}] Dictation canceled.", flush=True)
        print(f"[{config.APP_NAME}] Ready.", flush=True)

    def _poll_audio_level(self) -> None:
        """Polled on Qt main thread every 33 ms while recording to update UI state and waveform."""
        if not self.recorder.is_recording:
            return

        rms = self.recorder.latest_rms

        # Transition from LISTENING to SPEAKING once microphone volume exceeds threshold
        if self._current_state == PillState.LISTENING and rms >= config.SPEECH_THRESHOLD:
            self._current_state = PillState.SPEAKING
            self.overlay.set_state(PillState.SPEAKING)

        # Modulate waveform bar heights with live mic amplitude using noise gate & nonlinear curve
        if self._current_state in (PillState.LISTENING, PillState.SPEAKING):
            noise_floor = config.SPEECH_THRESHOLD * 0.5
            if rms <= noise_floor:
                level = 0.0
            else:
                # Map from noise floor to peak speech (approx 0.12 RMS)
                raw_norm = min(1.0, (rms - noise_floor) / max(1e-5, (0.12 - noise_floor)))
                # Perceptual power curve (gamma ~0.7) for responsive visual feedback
                level = raw_norm ** 0.7
            self.overlay.set_audio_level(level)

    def process_audio(self, audio: np.ndarray) -> None:
        """Transcribe audio on worker thread, apply corrections, and paste text at cursor."""
        with self._lock:
            self.is_processing = True

        try:
            text = self.transcriber.transcribe(audio)
            if not text:
                print(f"[{config.APP_NAME}] No speech detected.", flush=True)
            else:
                text = apply_corrections(text)
                print(f"[{config.APP_NAME}] Transcribed: {text}", flush=True)
                paste_text(text)
        finally:
            with self._lock:
                self.is_processing = False
                self._current_state = PillState.IDLE
            try:
                self.ui_bridge.state_changed.emit(PillState.IDLE)
            except RuntimeError:
                pass
            print(f"[{config.APP_NAME}] Ready.", flush=True)

    # -------------------------------------------------------------------------
    # Hotkey Event Callbacks (Executed on pynput background listener thread)
    # -------------------------------------------------------------------------

    def on_press(self, key: keyboard.Key | keyboard.KeyCode | None) -> None:
        """Handle Right Option key press with debounce and bridge to Qt main thread."""
        if key != self.target_hotkey:
            return

        now = time.monotonic()
        if now - self._last_hotkey_press_time < config.HOTKEY_DEBOUNCE_SECONDS:
            return
        self._last_hotkey_press_time = now

        # Emit signal to dispatch handler onto Qt GUI main thread safely
        self.ui_bridge.hotkey_toggled.emit()

    def on_release(self, key: keyboard.Key | keyboard.KeyCode | None) -> None:
        """Handle Right Option key release."""
        if config.HOTKEY_MODE == "toggle":
            # In toggle mode, releasing the key does not stop dictation
            return

        if key != self.target_hotkey:
            return
        self.ui_bridge.hotkey_toggled.emit()

    # -------------------------------------------------------------------------
    # Shutdown
    # -------------------------------------------------------------------------

    def stop_and_exit(self, app: QApplication) -> None:
        """Clean up audio, listener, and overlay, then exit the application."""
        print(f"\n[{config.APP_NAME}] Exiting cleanly...", flush=True)
        self._level_timer.stop()
        if self.recorder.is_recording:
            self.recorder.cancel()
        if self._listener is not None and self._listener.is_alive():
            self._listener.stop()
        self.overlay.close()
        app.quit()


def main() -> None:
    """Entry point for Spokn."""
    app = QApplication.instance() or QApplication(sys.argv)

    # Heartbeat timer so Python signal handlers can execute inside Qt event loop
    heartbeat_timer = QTimer()
    heartbeat_timer.timeout.connect(lambda: None)
    heartbeat_timer.start(200)

    dictation_app = DictationApp()
    signal.signal(signal.SIGINT, lambda *_: dictation_app.stop_and_exit(app))

    dictation_app.start()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
