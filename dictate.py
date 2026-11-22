"""
Spokn - Personal Offline Dictation for Apple Silicon

High-Level V1 Flow:
1. Startup & Warm-Up:
   - Load configuration and warm up the local MLX Whisper model in memory.
2. Hold-to-Talk Hotkey Listener:
   - Press and hold Right Option (HOTKEY) to begin recording audio from the microphone.
   - Stream audio chunks into an in-memory buffer.
3. Release & Transcribe:
   - Releasing the key stops recording.
   - If duration >= MIN_RECORDING_DURATION, pass the buffer to the local MLX Whisper model.
4. Paste & Restore:
   - Back up current clipboard content using pbpaste.
   - Copy transcribed text to the clipboard using pbcopy and simulate Cmd+V via osascript.
   - After CLIPBOARD_RESTORE_DELAY, restore the previous clipboard content.
"""

from __future__ import annotations

import subprocess
import threading
import time
import numpy as np
from pynput import keyboard
import sounddevice as sd
import mlx_whisper

import config


class AudioRecorder:
    """Handles real-time microphone audio capture and normalization using sounddevice."""

    def __init__(self) -> None:
        """Initializes internal audio chunk buffer, recording flag, and stream handle."""
        self._buffer: list[np.ndarray] = []
        self._is_recording: bool = False
        self._stream: sd.InputStream | None = None

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
        """Callback invoked by PortAudio on a dedicated audio thread for each block.

        Appends a copy of indata to the buffer while recording. Logs any stream
        status warnings (such as buffer overflow or underflow).
        """
        if status:
            print(f"[{config.APP_NAME}] Audio stream warning: {status}", flush=True)

        if self._is_recording:
            # indata is a view managed by sounddevice and gets overwritten;
            # store an explicit copy of each chunk.
            self._buffer.append(indata.copy())

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

        # Why normalize int16 to float32:
        # Whisper expects 32-bit floating point audio waveform samples normalized
        # to the [-1.0, 1.0] range. Dividing standard signed 16-bit integer PCM values
        # (-32768 to 32767) by 32768.0 converts to float32 without clipping or loss.
        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        return audio_float32


class Transcriber:
    """Manages local speech-to-text inference with mlx-whisper on Apple Silicon."""

    def __init__(self, model_repo: str = config.MODEL_REPO) -> None:
        """Initialize Transcriber with the configured Hugging Face / MLX model repo."""
        self.model_repo = model_repo

    def warm_up(self) -> None:
        """Warm up the MLX model and Apple Silicon unified memory before live dictation.

        Why run a silent warm-up at startup:
        1. Model Weights & Compilation: On first launch, huggingface_hub downloads
           the model weights if not already cached. MLX then compiles computation
           graphs and loads the neural weights into unified memory.
        2. Eliminates Cold-Start Latency Spike: The initial inference call typically
           incurs a 1-3 second warm-up delay. By passing a 1-second silent audio buffer
           at application startup, the very first user dictation transcribes with
           instant, responsive latency.
        """
        print(f"[{config.APP_NAME}] Loading model ({self.model_repo})...", flush=True)
        try:
            # 1 second of silent float32 audio at 16kHz
            silent_audio = np.zeros(config.SAMPLE_RATE, dtype=np.float32)
            mlx_whisper.transcribe(silent_audio, path_or_hf_repo=self.model_repo)
            print(f"[{config.APP_NAME}] Model ready.", flush=True)
        except Exception as exc:
            print(
                f"[{config.APP_NAME}] Warning during model warm-up: {exc}",
                flush=True,
            )

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe a normalized 1D float32 numpy audio buffer to text.

        Why passing float32 numpy array directly avoids disk I/O:
        Traditional transcription wrappers write audio chunks to temporary .wav files
        on disk, which introduces file system I/O latency, locks, and SSD write wear.
        mlx-whisper accepts an in-memory 1D float32 numpy array directly. Audio flows
        seamlessly from microphone memory buffers straight into Apple Silicon Metal/GPU
        unified memory with zero disk round-trips.

        Returns:
            str: Stripped transcribed text, or empty string on error or no speech.
        """
        try:
            result = mlx_whisper.transcribe(audio, path_or_hf_repo=self.model_repo)
            return result.get("text", "").strip()
        except Exception as exc:
            print(
                f"[{config.APP_NAME}] Transcription error: {exc}",
                flush=True,
            )
            return ""


def get_clipboard() -> str | None:
    """Read the current text content from the macOS system clipboard using pbpaste.

    Returns:
        str | None: Clipboard string content on success, or None on failure.
    """
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
    """Write text to the macOS system clipboard using pbcopy.

    Args:
        text: The string to place onto the clipboard.
    """
    subprocess.run(
        ["pbcopy"],
        input=text,
        text=True,
        check=True,
    )


def paste_text(text: str) -> bool:
    """Copy text to clipboard, simulate Cmd+V via osascript, and restore prior clipboard.

    Why wait CLIPBOARD_RESTORE_DELAY before restoring:
    Target macOS applications (Electron, browsers, terminals, text editors) process
    simulated Cmd+V asynchronously from their respective OS run-loops. If pbcopy
    restores the old clipboard content too quickly (e.g. immediately after keystroke),
    the target application might paste the previously restored clipboard content
    instead of the transcribed text. A short pause (config.CLIPBOARD_RESTORE_DELAY)
    guarantees the active application has consumed the transcribed text before
    the original clipboard is restored.

    Why paste errors are caught:
    Simulating keystrokes via AppleScript requires macOS Accessibility permissions
    (System Settings -> Privacy & Security -> Accessibility). If permissions are
    missing, revoked, or an OS error occurs, catching the exception ensures the
    background thread logs a clean diagnostic warning and DictationApp returns
    gracefully to Ready state without crashing the global hotkey listener.

    Returns:
        bool: True if paste succeeded, False otherwise.
    """
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
    """Resolve the configured hotkey string to a pynput keyboard.Key enum value.

    Returns:
        keyboard.Key: The resolved pynput Key enum attribute.

    Raises:
        ValueError: If config.HOTKEY is not a valid attribute on keyboard.Key.
    """
    hotkey = getattr(keyboard.Key, config.HOTKEY, None)
    if hotkey is None:
        raise ValueError(
            f"Invalid hotkey '{config.HOTKEY}' specified in config.py. "
            f"Expected a valid attribute of pynput.keyboard.Key (e.g. 'alt_r', 'ctrl_r')."
        )
    return hotkey


class DictationApp:
    """Coordinates hold-to-talk hotkey events, audio recording, and background processing.

    macOS Permissions & Architecture Notes:
    - Accessibility Permission: pynput monitors global key events via macOS Quartz
      event taps (CGEventTap), and osascript simulates Cmd+V keystrokes. Terminal
      apps must be granted Accessibility permissions under System Settings ->
      Privacy & Security -> Accessibility.
    - Key-Repeat Protection: When a physical key is held down, macOS fires repeated
      key-press events at the OS key-repeat rate. Guarding with `not recorder.is_recording`
      and `not is_processing` prevents key-repeat events from restarting or corrupting
      the recording while the user is simply holding down the key.
    - Asynchronous Processing: Heavy processing and MLX Whisper inference run on a
      background daemon thread (`process_audio`). This ensures the pynput listener
      event loop is never blocked, keeping hotkey detection immediate and fluid.
    """

    def __init__(
        self,
        recorder: AudioRecorder | None = None,
        transcriber: Transcriber | None = None,
    ) -> None:
        self.recorder = recorder or AudioRecorder()
        self.transcriber = transcriber or Transcriber()
        self.target_hotkey = get_hotkey()
        self.is_processing = False
        self._lock = threading.Lock()
        self._listener: keyboard.Listener | None = None

    def on_press(self, key: keyboard.Key | keyboard.KeyCode | None) -> None:
        """Handle hotkey press to start audio recording.

        Filters out key-repeat events and checks that no other transcription
        job is actively processing.
        """
        if key != self.target_hotkey:
            return

        with self._lock:
            # Prevent re-triggering while already recording (key-repeat) or processing
            if self.recorder.is_recording or self.is_processing:
                return

        if self.recorder.start():
            print(f"[{config.APP_NAME}] Listening...", flush=True)

    def on_release(self, key: keyboard.Key | keyboard.KeyCode | None) -> None:
        """Handle hotkey release to stop audio recording and dispatch processing."""
        if key != self.target_hotkey:
            return

        if not self.recorder.is_recording:
            return

        audio = self.recorder.stop()
        if audio is None:
            print(f"[{config.APP_NAME}] Ignored short recording.", flush=True)
            print(f"[{config.APP_NAME}] Ready.", flush=True)
            return

        print(f"[{config.APP_NAME}] Transcribing...", flush=True)

        # Dispatch audio processing off the pynput listener thread
        worker = threading.Thread(
            target=self.process_audio,
            args=(audio,),
            daemon=True,
            name="AudioProcessorThread",
        )
        worker.start()

    def process_audio(self, audio: np.ndarray) -> None:
        """Process captured audio on a background worker thread.

        Transcribes the normalized float32 buffer with mlx-whisper, prints
        the resulting text, and simulates Cmd+V paste at the active cursor.
        """
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
            print(f"[{config.APP_NAME}] Ready.", flush=True)

    def run(self) -> None:
        """Start the keyboard listener and wait for events until interrupted."""
        self.transcriber.warm_up()
        print(
            f"[{config.APP_NAME}] Ready. Hold Right Option to talk (Ctrl+C to quit).",
            flush=True,
        )

        with keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release,
        ) as listener:
            self._listener = listener
            try:
                # Use bounded join loop so KeyboardInterrupt (SIGINT) is caught immediately
                while listener.is_alive():
                    listener.join(0.5)
            except KeyboardInterrupt:
                pass
            finally:
                self.stop()

    def stop(self) -> None:
        """Safely clean up active recording and stop listener on shutdown."""
        if self.recorder.is_recording:
            self.recorder.stop()
        if self._listener is not None and self._listener.is_alive():
            self._listener.stop()
        print(f"\n[{config.APP_NAME}] Exiting...", flush=True)


def main() -> None:
    """Entry point for Spokn."""
    app = DictationApp()
    try:
        app.run()
    except KeyboardInterrupt:
        app.stop()


if __name__ == "__main__":
    main()
