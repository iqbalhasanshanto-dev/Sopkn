# Spokn 🎙️

**Spokn** is a personal, ultra-fast, 100% offline hold-to-talk macOS dictation tool (Wispr Flow-style) built specifically for Apple Silicon (optimized for M-series chips such as MacBook Air M4 with 16GB RAM).

It uses Apple's [MLX](https://github.com/ml-explore/mlx) framework via `mlx-whisper` for local on-device speech-to-text inference with zero cloud dependencies, zero subscriptions, and complete privacy.

---

## Features

- **100% Offline & Private**: Audio never leaves your Mac. Zero external API calls or telemetry.
- **Apple Silicon Optimized**: Leverages Apple Silicon unified memory and GPU/Neural Engine via MLX.
- **Hold-to-Talk (Wispr Flow Style)**: Hold `Right Option` to speak, release to transcribe and paste.
- **Non-Destructive Clipboard**: Copies and pastes the transcript at the cursor (`Cmd+V`), then restores your prior clipboard content.

---

## System Prerequisites

Install the required native libraries using [Homebrew](https://brew.sh/):

```bash
brew install portaudio ffmpeg
```

- `portaudio`: Required by `sounddevice` for microphone audio capture.
- `ffmpeg`: Required by audio processing tools and Whisper for audio decoding/encoding.

---

## Installation & Setup

1. **Clone or navigate to the repository**:
   ```bash
   cd Spokn
   ```

2. **Create and activate a Python virtual environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Required macOS Permissions

Spokn interacts with your microphone and global system events. You must grant permissions in **System Settings → Privacy & Security**:

1. **Microphone**:
   - Navigate to **System Settings → Privacy & Security → Microphone**.
   - Enable access for your terminal application (e.g., Terminal, iTerm2, Ghostty, VS Code) or the packaged application.
2. **Accessibility**:
   - Navigate to **System Settings → Privacy & Security → Accessibility**.
   - Enable access for your terminal application so `pynput` can monitor global hotkeys (Right Option) and simulate keystrokes (`Cmd+V` paste).

---

## Configuration

All customizable settings are located in [`config.py`](file:///Users/shanto/Developer/Spokn/config.py):

| Setting | Default | Description |
| :--- | :--- | :--- |
| `APP_NAME` | `"Spokn"` | Name of the application |
| `MODEL_REPO` | `"mlx-community/whisper-small.en-mlx"` | Local MLX model repo (Optional upgrade: `"mlx-community/whisper-large-v3-turbo"`) |
| `HOTKEY` | `"alt_r"` | Hold-to-talk trigger (Right Option key) |
| `SAMPLE_RATE` | `16000` | Audio sample rate (16kHz required by Whisper) |
| `CHANNELS` | `1` | Audio channels (mono) |
| `DTYPE` | `"int16"` | Audio buffer data type |
| `MIN_RECORDING_DURATION` | `0.3` | Minimum seconds required to process audio |
| `CLIPBOARD_RESTORE_DELAY` | `0.15` | Seconds before restoring previous clipboard content |

---

## Custom Auto-Corrections

You can customize domain-specific word or phrase replacements by copying [`corrections.example.json`](file:///Users/shanto/Developer/Spokn/corrections.example.json) to `corrections.json` (gitignored for privacy):

```bash
cp corrections.example.json corrections.json
```

Add your personal replacements to `corrections.json`:
```json
{
  "pie side": "PySide6",
  "code forces": "Codeforces"
}
```
Replacements are reloaded dynamically on each transcription without needing to restart Spokn.

---

## Running Spokn

Run the entry point from your virtual environment:

```bash
python dictate.py
```
