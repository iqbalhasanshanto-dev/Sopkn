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


# 10 speech-range frequency bands spanning ~120 Hz to 3800 Hz
SPEECH_BAND_EDGES = [
    120.0, 220.0, 360.0, 550.0, 800.0, 1200.0, 1700.0, 2300.0, 2900.0, 3400.0, 3800.0
]


class AudioRecorder:
    """Handles real-time microphone audio capture and normalization using sounddevice.

    Thread Safety:
    - Never calls Qt functions or emits Qt signals from _audio_callback (the PortAudio
      real-time audio thread).
    - Computes peak-hold RMS volume and a 10-band speech spectrum, storing them
      atomically for the Qt main thread to poll safely via consume_audio_levels().
    """

    def __init__(self) -> None:
        """Initializes internal audio chunk buffer, recording flag, and stream handle."""
        self._buffer: list[np.ndarray] = []
        self._is_recording: bool = False
        self._stream: sd.InputStream | None = None
        self._peak_rms: float = 0.0
        self._peak_bands: list[float] = [0.0] * 10

    @property
    def is_recording(self) -> bool:
        """Return whether audio is currently actively recording."""
        return self._is_recording

    @property
    def latest_rms(self) -> float:
        """Return the current peak RMS volume since the last consumption."""
        return self._peak_rms

    def consume_audio_levels(self) -> tuple[float, list[float]]:
        """Atomically read and reset peak RMS and 10-band spectrum since the last poll."""
        rms = self._peak_rms
        bands = list(self._peak_bands)
        self._peak_rms = 0.0
        self._peak_bands = [0.0] * 10
        return rms, bands

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

            # Convert int16 audio to float32 in range [-1.0, 1.0]
            data_float = indata.astype(np.float32).flatten() / 32768.0
            n_samples = len(data_float)
            if n_samples == 0:
                return

            # Compute RMS amplitude of this chunk and update peak-hold
            chunk_rms = float(np.sqrt(np.mean(data_float ** 2)))
            if chunk_rms > self._peak_rms:
                self._peak_rms = chunk_rms

            # Compute 10-band speech spectrum (~120 Hz to 3800 Hz) via rfft
            rfft_mag = np.abs(np.fft.rfft(data_float))
            freqs = np.fft.rfftfreq(n_samples, d=1.0 / config.SAMPLE_RATE)
            fft_norm = rfft_mag / (n_samples / 2.0)

            for i in range(10):
                f_low = SPEECH_BAND_EDGES[i]
                f_high = SPEECH_BAND_EDGES[i + 1]
                mask = (freqs >= f_low) & (freqs < f_high)
                if np.any(mask):
                    val = float(np.mean(fft_norm[mask]))
                else:
                    idx = int(np.argmin(np.abs(freqs - (f_low + f_high) / 2.0)))
                    val = float(fft_norm[idx])
                if val > self._peak_bands[i]:
                    self._peak_bands[i] = val

    def start(self) -> bool:
        """Reset buffer and open/start the sounddevice InputStream."""
        if self._is_recording:
            return True

        self._buffer.clear()
        self._peak_rms = 0.0
        self._peak_bands = [0.0] * 10
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
        self._peak_rms = 0.0
        self._peak_bands = [0.0] * 10

        if self._stream is not None:
            try:
                self._stream.stop()
            except Exception as exc:
                print(f"[{config.APP_NAME}] Warning stopping audio stream: {exc}", flush=True)
                try:
                    self._stream.abort()
                except Exception:
                    pass
            try:
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
        self._peak_rms = 0.0
        self._peak_bands = [0.0] * 10
        if self._stream is not None:
            try:
                self._stream.stop()
            except Exception as exc:
                print(f"[{config.APP_NAME}] Warning stopping audio stream: {exc}", flush=True)
                try:
                    self._stream.abort()
                except Exception:
                    pass
            try:
                self._stream.close()
            except Exception as exc:
                print(f"[{config.APP_NAME}] Warning closing audio stream: {exc}", flush=True)
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


def is_target_hotkey(key: keyboard.Key | keyboard.KeyCode | None) -> bool:
    """Return True if the pressed key matches the configured hotkey.

    Handles special macOS keycodes:
    - cmd_r: Key.cmd_r or virtual keycode 54 (Right Command).
    - f5: Key.f5 or virtual keycodes 96 (standard F5) and 176 (Apple Silicon Mic key).
    """
    if key is None:
        return False

    if config.HOTKEY == "cmd_r":
        return key == keyboard.Key.cmd_r or getattr(key, "vk", None) == 54

    if config.HOTKEY == "f5":
        return key == keyboard.Key.f5 or getattr(key, "vk", None) in (96, 176)

    target = getattr(keyboard.Key, config.HOTKEY, None)
    return key == target if target is not None else False


def get_hotkey() -> keyboard.Key:
    """Resolve the configured hotkey string to a pynput keyboard.Key enum value."""
    hotkey = getattr(keyboard.Key, config.HOTKEY, None)
    if hotkey is None:
        raise ValueError(
            f"Invalid hotkey '{config.HOTKEY}' specified in config.py. "
            f"Expected a valid attribute of pynput.keyboard.Key (e.g. 'f5', 'alt_r', 'ctrl_r')."
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

        # Double-tap tracking state
        self._last_tap_time: float = 0.0
        self._other_key_pressed: bool = False
        self._hotkey_held: bool = False

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
                f"[{config.APP_NAME}] Ready. Double-tap Right Command or click pill to talk (Ctrl+C to quit).",
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
        self.overlay.set_audio_bands([0.0] * 10)
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
        self.overlay.set_audio_bands([0.0] * 10)
        self.overlay.set_audio_level(0.0)
        self.recorder.cancel()
        self._current_state = PillState.IDLE
        self.overlay.set_state(PillState.IDLE)
        print(f"[{config.APP_NAME}] Dictation canceled.", flush=True)
        print(f"[{config.APP_NAME}] Ready.", flush=True)

    def _poll_audio_level(self) -> None:
        """Polled on Qt main thread every 33 ms while recording to update UI state and vocal waveform."""
        if not self.recorder.is_recording:
            return

        rms, raw_bands = self.recorder.consume_audio_levels()

        # Transition from LISTENING to SPEAKING once microphone volume exceeds threshold
        if self._current_state == PillState.LISTENING and rms >= config.SPEECH_THRESHOLD:
            self._current_state = PillState.SPEAKING
            self.overlay.set_state(PillState.SPEAKING)

        # Modulate 10-bar waveform with calibrated MacBook Air mic sensitivity
        if self._current_state in (PillState.LISTENING, PillState.SPEAKING):
            NOISE_FLOOR = 0.002
            PEAK_SPEECH = 0.035

            if rms <= NOISE_FLOOR:
                scaled_bands = [0.0] * 10
            else:
                raw_norm = min(1.0, max(0.0, (rms - NOISE_FLOOR) / (PEAK_SPEECH - NOISE_FLOOR)))
                speech_level = raw_norm ** 0.6

                max_band = max(raw_bands) if raw_bands else 0.0
                if max_band > 1e-6:
                    # Spectral equalization tilt across speech frequencies (formants to sibilance)
                    tilt = [0.9, 0.95, 1.0, 1.05, 1.15, 1.25, 1.35, 1.45, 1.55, 1.65]
                    scaled_bands = []
                    for i, b in enumerate(raw_bands):
                        norm_b = min(1.0, (b / max_band) * tilt[i])
                        band_val = speech_level * (0.35 + 0.65 * norm_b)
                        scaled_bands.append(min(1.0, max(0.0, band_val)))
                else:
                    scaled_bands = [speech_level] * 10

            self.overlay.set_audio_bands(scaled_bands)

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
        """Handle key press event from pynput listener."""
        is_hotkey = is_target_hotkey(key)

        if not is_hotkey:
            # If any other key is pressed (e.g. 'c' in Cmd+C), invalidate double-tap
            self._other_key_pressed = True
            self._last_tap_time = 0.0
            return

        # Target hotkey pressed
        if not self._hotkey_held:
            # Fresh press (not repeat while holding)
            self._hotkey_held = True
            self._other_key_pressed = False

        if config.HOTKEY_MODE == "toggle":
            now = time.monotonic()
            if now - self._last_hotkey_press_time >= config.HOTKEY_DEBOUNCE_SECONDS:
                self._last_hotkey_press_time = now
                self.ui_bridge.hotkey_toggled.emit()

    def on_release(self, key: keyboard.Key | keyboard.KeyCode | None) -> None:
        """Handle key release event from pynput listener."""
        is_hotkey = is_target_hotkey(key)
        if not is_hotkey:
            return

        self._hotkey_held = False

        if config.HOTKEY_MODE == "double_tap":
            if not self._other_key_pressed:
                now = time.monotonic()
                if now - self._last_tap_time <= config.DOUBLE_TAP_WINDOW:
                    self._last_tap_time = 0.0
                    self.ui_bridge.hotkey_toggled.emit()
                else:
                    self._last_tap_time = now
        elif config.HOTKEY_MODE == "hold":
            self.ui_bridge.hotkey_toggled.emit()

    # -------------------------------------------------------------------------
    # Shutdown
    # -------------------------------------------------------------------------

    def stop_and_exit(self, app: QApplication) -> None:
        """Clean up audio, listener, and overlay, then exit the application."""
        print(f"\n[{config.APP_NAME}] Exiting cleanly...", flush=True)
        self._level_timer.stop()
        self.overlay.set_audio_bands([0.0] * 10)
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
