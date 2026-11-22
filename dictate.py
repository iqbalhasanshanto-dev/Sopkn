"""
Spokn - Personal Offline Dictation for Apple Silicon

Architecture & V1 Flow:
1. UI Initialization & Model Warm-Up:
   - Initialize PySide6 QApplication and display the DictationPillOverlay in IDLE state.
   - Run MLX Whisper warm-up asynchronously in a background thread so the UI appears instantly.
2. Dual Trigger Dictation (Hold-to-Talk or Click-to-Toggle):
   - Hold Right Option (HOTKEY) or click the IDLE pill to start microphone recording (LISTENING).
   - Once microphone RMS exceeds SPEECH_THRESHOLD, transition to SPEAKING with live waveform modulation.
3. Release or Stop:
   - Releasing the hotkey or clicking Stop transitions to COLLAPSING -> PROCESSING.
   - Clicking × Cancel at any time cancels recording and resets directly to IDLE.
4. Local Inference & Paste:
   - In-memory float32 audio is transcribed with mlx-whisper (conditioned on language & vocabulary hints).
   - Copies transcript to clipboard, simulates Cmd+V via osascript, and restores prior clipboard.
"""

from __future__ import annotations

from collections.abc import Callable
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

    state_changed = Signal(object)  # PillState
    audio_level = Signal(float)     # float (0.0 to 1.0)


class AudioRecorder:
    """Handles real-time microphone audio capture and normalization using sounddevice."""

    def __init__(self, level_callback: Callable[[float], None] | None = None) -> None:
        """Initializes internal audio chunk buffer, recording flag, and stream handle."""
        self._buffer: list[np.ndarray] = []
        self._is_recording: bool = False
        self._stream: sd.InputStream | None = None
        self._level_callback = level_callback

    @property
    def is_recording(self) -> bool:
        """Return whether audio is currently actively recording."""
        return self._is_recording

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

            if self._level_callback is not None:
                # Compute normalized RMS amplitude (0.0 to 1.0) from int16 audio
                data_float = indata.astype(np.float32) / 32768.0
                rms = float(np.sqrt(np.mean(data_float ** 2)))
                self._level_callback(rms)

    def start(self) -> bool:
        """Reset buffer and open/start the sounddevice InputStream.

        Returns:
            bool: True if recording started successfully, False otherwise.
        """
        if self._is_recording:
            return True

        self._buffer.clear()
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
        """Stop and close the audio stream, then concatenate and normalize captured audio.

        Returns:
            np.ndarray | None: 1D float32 array normalized to [-1.0, 1.0] ready
                for mlx-whisper, or None if no audio was captured or if the
                recording was shorter than config.MIN_RECORDING_DURATION.
        """
        self._is_recording = False

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

        # Thread-safe Qt signal bridge
        self.ui_bridge = UiBridge()
        self.ui_bridge.state_changed.connect(self.overlay.set_state)
        self.ui_bridge.audio_level.connect(self.overlay.set_audio_level)

        # Connect overlay interactive mouse events
        self.overlay.idle_clicked.connect(self.on_overlay_idle_clicked)
        self.overlay.cancel_clicked.connect(self.on_overlay_cancel_clicked)
        self.overlay.stop_clicked.connect(self.on_overlay_stop_clicked)

        # Audio recorder with live RMS volume callback
        self.recorder = AudioRecorder(level_callback=self._on_audio_chunk)

        # State tracking
        self.is_ready = False
        self.is_processing = False
        self._current_state = PillState.IDLE
        self._lock = threading.Lock()
        self._listener: keyboard.Listener | None = None

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
                f"[{config.APP_NAME}] Ready. Hold Right Option or click pill to talk (Ctrl+C to quit).",
                flush=True,
            )

        warmup_thread = threading.Thread(
            target=_warmup_worker,
            daemon=True,
            name="ModelWarmupThread",
        )
        warmup_thread.start()

    # -------------------------------------------------------------------------
    # Dictation Control Logic
    # -------------------------------------------------------------------------

    def start_dictation(self) -> None:
        """Start audio recording and transition UI to LISTENING state."""
        with self._lock:
            if not self.is_ready or self.recorder.is_recording or self.is_processing:
                return

        if self.recorder.start():
            self._current_state = PillState.LISTENING
            self.ui_bridge.state_changed.emit(PillState.LISTENING)
            print(f"[{config.APP_NAME}] Listening...", flush=True)

    def stop_dictation(self) -> None:
        """Stop recording, trigger COLLAPSING -> PROCESSING, and transcribe."""
        if not self.recorder.is_recording:
            return

        audio = self.recorder.stop()
        if audio is None:
            self._current_state = PillState.IDLE
            self.ui_bridge.state_changed.emit(PillState.IDLE)
            print(f"[{config.APP_NAME}] Ignored short recording.", flush=True)
            print(f"[{config.APP_NAME}] Ready.", flush=True)
            return

        # Transition to COLLAPSING (which auto-morphs to PROCESSING after 220 ms)
        self._current_state = PillState.PROCESSING
        self.ui_bridge.state_changed.emit(PillState.COLLAPSING)
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
        self.recorder.cancel()
        self._current_state = PillState.IDLE
        self.ui_bridge.state_changed.emit(PillState.IDLE)
        print(f"[{config.APP_NAME}] Dictation canceled.", flush=True)
        print(f"[{config.APP_NAME}] Ready.", flush=True)

    def _on_audio_chunk(self, rms: float) -> None:
        """Process live audio volume: switch to SPEAKING on speech, and modulate bars."""
        if not self.recorder.is_recording:
            return

        # If in LISTENING and volume crosses speech threshold, morph to SPEAKING
        if self._current_state == PillState.LISTENING and rms >= config.SPEECH_THRESHOLD:
            self._current_state = PillState.SPEAKING
            self.ui_bridge.state_changed.emit(PillState.SPEAKING)

        # Modulate waveform bar heights with live mic amplitude
        if self._current_state == PillState.SPEAKING:
            level = min(1.0, rms * 10.0)
            self.ui_bridge.audio_level.emit(level)

    def process_audio(self, audio: np.ndarray) -> None:
        """Transcribe audio on worker thread and paste text at cursor."""
        with self._lock:
            self.is_processing = True

        try:
            text = self.transcriber.transcribe(audio)
            if not text:
                print(f"[{config.APP_NAME}] No speech detected.", flush=True)
            else:
                print(f"[{config.APP_NAME}] Transcribed: {text}", flush=True)
                paste_text(text)
        finally:
            with self._lock:
                self.is_processing = False
                self._current_state = PillState.IDLE
            self.ui_bridge.state_changed.emit(PillState.IDLE)
            print(f"[{config.APP_NAME}] Ready.", flush=True)

    # -------------------------------------------------------------------------
    # Hotkey Event Callbacks
    # -------------------------------------------------------------------------

    def on_press(self, key: keyboard.Key | keyboard.KeyCode | None) -> None:
        """Handle Right Option key press (filtered for key-repeat)."""
        if key != self.target_hotkey:
            return
        self.start_dictation()

    def on_release(self, key: keyboard.Key | keyboard.KeyCode | None) -> None:
        """Handle Right Option key release."""
        if key != self.target_hotkey:
            return
        self.stop_dictation()

    # -------------------------------------------------------------------------
    # Overlay Mouse Event Callbacks
    # -------------------------------------------------------------------------

    def on_overlay_idle_clicked(self) -> None:
        """Start dictation when the resting IDLE pill is clicked."""
        self.start_dictation()

    def on_overlay_stop_clicked(self) -> None:
        """Stop dictation and transcribe when the red Stop button is clicked."""
        self.stop_dictation()

    def on_overlay_cancel_clicked(self) -> None:
        """Discard audio when the × Cancel button is clicked."""
        self.cancel_dictation()

    # -------------------------------------------------------------------------
    # Shutdown
    # -------------------------------------------------------------------------

    def stop_and_exit(self, app: QApplication) -> None:
        """Clean up audio, listener, and overlay, then exit the application."""
        print(f"\n[{config.APP_NAME}] Exiting cleanly...", flush=True)
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
