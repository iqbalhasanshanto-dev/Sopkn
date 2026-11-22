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

import config


def main() -> None:
    """Entry point stub for Spokn."""
    print(f"{config.APP_NAME} initialized. Ready to wire audio and hotkey modules.")


if __name__ == "__main__":
    main()
