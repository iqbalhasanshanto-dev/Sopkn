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
   - Back up current clipboard content.
   - Copy transcribed text to the clipboard and simulate Cmd+V via pynput.
   - After CLIPBOARD_RESTORE_DELAY, restore the previous clipboard content.
"""

from __future__ import annotations

import time
import numpy as np
import sounddevice as sd

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


def main() -> None:
    """Temporary self-test verifying microphone capture, portaudio, and permissions."""
    print(f"{config.APP_NAME} initialized. Ready to wire audio and hotkey modules.")
    print("Testing microphone capture (2-second self-test recording)...")

    recorder = AudioRecorder()
    if not recorder.start():
        print("Microphone self-test failed: could not open audio stream.")
        return

    print("Recording... Speak into your microphone!")
    time.sleep(2)

    audio = recorder.stop()
    if audio is not None:
        duration = len(audio) / config.SAMPLE_RATE
        print("Test recording successful!")
        print(f"  Duration: {duration:.2f}s")
        print(f"  Shape:    {audio.shape}")
        print(f"  Dtype:    {audio.dtype}")
        print(f"  Range:    [{audio.min():.4f}, {audio.max():.4f}]")
    else:
        print("No valid audio captured (recording too short or empty).")


if __name__ == "__main__":
    main()
