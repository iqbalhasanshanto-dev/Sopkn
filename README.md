<div align="center">
  <img src="./assets/app-icon.png" alt="Spokn Logo" width="108" />
  <h1>Spokn</h1>
  <p><em>Offline, zero-cost macOS dictation built for Apple Silicon</em></p>

  <p>
    <a href="https://github.com/iqbalhasanshanto/Spokn"><img src="https://img.shields.io/badge/version-1.1.0-blue.svg" alt="Version 1.1.0" /></a>
    <a href="https://apple.com"><img src="https://img.shields.io/badge/platform-macOS%20Apple%20Silicon-lightgrey.svg" alt="macOS Apple Silicon" /></a>
    <a href="https://python.org"><img src="https://img.shields.io/badge/python-3.11%2B-blue.svg" alt="Python 3.11+" /></a>
    <a href="https://github.com/ml-explore/mlx"><img src="https://img.shields.io/badge/engine-MLX%20Whisper-orange.svg" alt="MLX Whisper" /></a>
    <img src="https://img.shields.io/badge/privacy-100%25%20Offline-success.svg" alt="100% Offline" />
  </p>

  <p>
    <a href="#overview--key-features">Features</a> •
    <a href="#how-the-dictation-pill-works">How It Works</a> •
    <a href="#tech-stack">Tech Stack</a> •
    <a href="#quick-start">Quick Start</a> •
    <a href="#personal-dictionary--optional-ollama-cleanup">Personal Dictionary</a> •
    <a href="#configuration">Configuration</a> •
    <a href="#project-structure">Project Structure</a>
  </p>
</div>

---

## Overview & Key Features

Spokn is a personal, ultra-responsive macOS dictation tool built from the ground up for Apple Silicon. It brings a refined hold-to-talk / double-tap workflow to macOS without cloud dependencies, API keys, or recurring subscription fees.

- **100% Offline on Apple Silicon**: Runs OpenAI's `whisper-large-v3-turbo` model locally via Apple's [MLX framework](https://github.com/ml-explore/mlx). Audio data never leaves unified memory, ensuring zero cloud latency, complete privacy, and no word or time limits.
- **Non-Activating 60 FPS Dictation Pill**: Anchored unobtrusively at the bottom of your primary display with translucent obsidian glass, subtle white rim glow, and a live vocal waveform rippling symmetrically from the center outward. Implemented as an `NSWindowStyleMaskNonactivatingPanel` so clicking never steals cursor focus or active window state from your editor, browser, or terminal.
- **Conflict-Free Double-Tap Hotkey**: Double-tap the **Right Command key (`⌘`)** to toggle dictation instantly without clashing with macOS Dictation (`F5`), Gemini shortcuts (`Fn`), or text-editing combos (`Cmd+C`, `Cmd+V`, `Cmd+Tab`).
- **Focus-Safe Mouse Controls & Pill Menu**: Single-click the resting pill to talk, click **Stop** to transcribe, or click **× Cancel** to discard audio. Right-click the pill anytime to toggle Ollama cleanup, restart the process, or quit cleanly.
- **Smart Spacing on Paste**: Transcribed text is copied to the system clipboard, injected at your active cursor via native `Cmd+V` simulation, and prior clipboard contents are restored automatically. Automatically appends a trailing space to prevent consecutive dictations from mashing words together.
- **Silent Background App (`LSUIElement`)**: Runs completely in the background with no Dock icon and no Terminal window. Ships with a self-contained installer (`install_app.py`) that sets up `/Applications/Spokn.app` and an optional LaunchAgent for automated run-at-login.

---

## How the Dictation Pill Works

The floating capsule morphs fluidly across 5 sequential visual states driven by a 220 ms spring animation:

| State | Canvas Size | Visuals & Behavior | Transition Trigger |
| :--- | :--- | :--- | :--- |
| **Idle** | `36 × 6 px` | Translucent obsidian resting bar with a 35% white border stroke. | Double-tap `⌘` or click the pill to start recording. Right-click opens the menu. |
| **Listening** | `124 × 36 px` | Expands with spring physics; reveals **× Cancel**, 10 resting waveform dots (2 px), and coral **Stop** button. | Activated immediately when recording begins. |
| **Speaking** | `124 × 36 px` | Live 60 FPS vocal waveform rippling symmetrically from center bars (4 & 5) outward to edges. | Transitions automatically when microphone volume exceeds speech threshold (`0.008`). |
| **Collapsing** | `100 × 28 px` | Waveform flattens and edge controls smoothly shrink into 4 px minimalist dots over 220 ms. | Triggered when recording stops (double-tap `⌘` or click Stop). |
| **Processing** | `90 × 26 px` | Compact pill displaying 3 sequential pulsing dots while MLX Whisper decodes speech and pastes text. | Automatically follows collapse; returns to **Idle** immediately after text is pasted. |

---

## Tech Stack

| Layer | Technology | Purpose |
| :--- | :--- | :--- |
| **Speech-to-Text** | [`mlx-whisper`](https://github.com/ml-explore/mlx-examples/tree/main/whisper) | Fast local inference with `whisper-large-v3-turbo` on Apple Silicon unified memory |
| **Audio Capture** | `sounddevice` + `numpy` | Low-latency 16 kHz mono 16-bit PCM audio stream with real-time RMS spectrum calculation |
| **Overlay UI** | `PySide6` (Qt6) + `AppKit` | Non-activating translucent desktop capsule, 60 FPS waveform, and spring morph animations |
| **Global Hotkey** | `pynput` | Background keyboard listener with double-tap detection and shortcut suppression |
| **Clipboard & Paste** | `pbcopy` / `pbpaste` + `osascript` | Non-destructive clipboard injection with automated trailing spacing |
| **Optional LLM** | [Ollama](https://ollama.com/) (`llama3.2:3b`) | Local grammar correction, filler word removal ("um", "uh"), and stutter smoothing |
| **macOS Packaging** | `install_app.py` | Standalone `/Applications/Spokn.app` bundler, `.icns` squircle generator, and login `LaunchAgent` |

---

## Quick Start

### 1. System Prerequisites

Install the required audio and multimedia libraries via [Homebrew](https://brew.sh/):

```bash
brew install portaudio ffmpeg
```

### 2. Clone the Repository

```bash
git clone https://github.com/iqbalhasanshanto/Spokn.git
cd Spokn
```

### 3. Create Virtual Environment & Install Dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 4. Build & Install `/Applications/Spokn.app`

Run the installer to package and install Spokn directly into your `/Applications` directory, and optionally register it to launch automatically whenever you log into macOS:

```bash
python install_app.py --autostart
```

*(To remove the autostart LaunchAgent at any time, run: `python install_app.py --remove-autostart`)*

### 5. Grant macOS Permissions

Open **System Settings → Privacy & Security** and grant access:
- **Microphone**: Allow `Spokn` (and your terminal application / `.venv/bin/python` if testing from CLI).
- **Accessibility**: Allow `Spokn` (and your terminal application / `.venv/bin/python` if testing from CLI) so Spokn can monitor global Right Command double-taps and paste transcribed text via `Cmd+V`.

### 6. Launch & Dictate

Launch Spokn from Spotlight (`Cmd + Space` → `Spokn`) or terminal:

```bash
open /Applications/Spokn.app
```

- **Double-tap Right Command (`cmd_r`)** or single-click the resting pill to start speaking.
- Double-tap again (or click the coral Stop button) to stop; Spokn transcribes on-device and automatically pastes text at your active cursor.
- Click **× Cancel** on the pill anytime to discard audio without transcribing.
- Right-click the pill to access controls (**Ollama Cleanup**, **Restart Spokn**, **Quit Spokn**).
- To monitor live transcription logs:
  ```bash
  tail -f ~/Library/Logs/Spokn.log
  ```

*(To run directly in terminal for debugging, simply run `python dictate.py`)*

---

## Personal Dictionary (`corrections.json`) & Optional Ollama Cleanup

### 1. Teaching Spokn Personal Vocabulary (`corrections.json`)

Whisper can occasionally mishear uncommon names, usernames, acronyms, or specialized technical terms. Spokn includes a built-in whole-word regex replacement engine that checks a local `corrections.json` file.

To set up your dictionary, copy the included example file:

```bash
cp corrections.example.json corrections.json
```

Add your personal substitutions to `corrections.json`:

```json
{
  "pie side": "PySide6",
  "code forces": "Codeforces",
  "anti gravity": "Antigravity",
  "shanto": "Shanto"
}
```

- **100% Private (Gitignored)**: `corrections.json` is listed in `.gitignore` so your personal vocabulary, names, and custom terms are never pushed to GitHub.
- **Hot-Reloading**: The dictionary is re-read from disk on every dictation. You can add or edit corrections anytime without restarting Spokn.

### 2. Optional Local Ollama Post-Processing

Spokn can optionally pass transcribed text through a local LLM via [Ollama](https://ollama.com/) to polish punctuation, capitalize sentences, remove verbal filler words (*"um"*, *"uh"*, *"you know"*), and smooth stutters before pasting.

1. Install Ollama and pull the lightweight Llama 3.2 model:
   ```bash
   ollama pull llama3.2:3b
   ```
2. Toggle **Ollama Cleanup** by right-clicking the floating Dictation Pill (or set `USE_OLLAMA_CLEANUP = True` in `config.py`).
3. If Ollama is not running or takes longer than `1.5s`, Spokn immediately falls back to raw Whisper output without blocking the UI.

---

## Configuration

All configurable parameters are centrally defined in [`config.py`](file:///Users/shanto/Developer/Spokn/config.py):

| Parameter | Current Default | Description |
| :--- | :--- | :--- |
| `APP_NAME` | `"Spokn"` | Application display name |
| `APP_VERSION` | `"1.1.0"` | Current application release version |
| `MODEL_REPO` | `"mlx-community/whisper-large-v3-turbo"` | Local MLX model repo optimized for Apple Silicon unified memory |
| `LANGUAGE` | `"en"` | Language code pinned to English to prevent accent hallucination or drift |
| `INITIAL_PROMPT` | *(Technical proper nouns)* | Vocabulary hints conditioning the tokenizer on domain-specific terms |
| `HOTKEY` | `"cmd_r"` | Right Command key (`cmd_r`). Alternatives: `"f5"` (Mic key), `"alt_r"` |
| `HOTKEY_MODE` | `"double_tap"` | Triggering mode (`"double_tap"`, `"toggle"`, or `"hold"`) |
| `DOUBLE_TAP_WINDOW` | `0.35` | Maximum seconds between two taps to register a double-tap trigger |
| `HOTKEY_DEBOUNCE_SECONDS` | `0.35` | Debounce window in seconds to ignore physical key chatter |
| `SAMPLE_RATE` | `16000` | Audio sample rate (16 kHz required by Whisper) |
| `CHANNELS` | `1` | Mono audio capture |
| `DTYPE` | `"int16"` | 16-bit PCM integer recording format |
| `SPEECH_THRESHOLD` | `0.008` | RMS threshold to transition pill from Listening to Speaking |
| `MIN_RECORDING_DURATION` | `0.3` | Minimum seconds of audio required before transcribing |
| `CLIPBOARD_RESTORE_DELAY` | `0.15` | Seconds after `Cmd+V` simulation before restoring previous clipboard |
| `TRAILING_SPACE` | `True` | Appends a trailing space if text does not already end in whitespace |
| `USE_OLLAMA_CLEANUP` | `False` | Optional local Ollama grammar/filler cleanup toggle |
| `OLLAMA_MODEL` | `"llama3.2:3b"` | Local Ollama model tag |
| `OLLAMA_URL` | `"http://localhost:11434/api/generate"` | Local Ollama generate endpoint |
| `OLLAMA_TIMEOUT` | `1.5` | Timeout (seconds) before falling back to raw Whisper text |

---

## Project Structure

Spokn is architected with a clean, modular structure:

```
Spokn/
├── assets/
│   └── app-icon.png         # Rendered 256x256 macOS squircle icon
├── config.py                # Central application configuration & thresholds
├── dictate.py               # Dictation coordinator, audio recorder & MLX transcriber
├── overlay.py               # Non-activating floating PySide6 Dictation Pill overlay
├── install_app.py           # macOS .app bundle & launch-at-login installer
├── corrections.json         # Personal word replacements (local, gitignored)
├── corrections.example.json # Generic example dictionary template
├── requirements.txt         # Python package dependencies
└── logo.png                 # Spokn vector source logo asset
```

---

## License

This project is open-source software licensed under the [MIT License](LICENSE). Built for Apple Silicon.
