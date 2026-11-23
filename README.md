<div align="center">
  <img src="./logo.png" alt="Spokn Logo" width="96" />
  <h1>Spokn</h1>
  <p><em>Offline, zero-cost macOS dictation built for Apple Silicon</em></p>

  <p>
    <a href="https://github.com/shantodev/Spokn"><img src="https://img.shields.io/badge/version-1.1.0-blue.svg" alt="Version 1.1.0" /></a>
    <a href="https://apple.com"><img src="https://img.shields.io/badge/platform-macOS%20Apple%20Silicon-lightgrey.svg" alt="macOS Apple Silicon" /></a>
    <a href="https://github.com/ml-explore/mlx"><img src="https://img.shields.io/badge/engine-MLX%20Whisper-orange.svg" alt="MLX Whisper" /></a>
    <img src="https://img.shields.io/badge/privacy-100%25%20Offline-success.svg" alt="100% Offline" />
  </p>
</div>

---

## Why Spokn?

- **100% Offline on Apple Silicon**: Uses Apple's [`mlx-whisper`](https://github.com/ml-explore/mlx) framework with `whisper-large-v3-turbo`. Audio never leaves unified memory—zero cloud dependencies, zero subscription costs, and no word or time limits.
- **Floating Non-Activating Dictation Pill**: Anchored unobtrusively at the bottom of your screen with an obsidian translucent glass blur, subtle white rim glow, and a silky 60 FPS vocal waveform that ripples symmetrically from center outward. Never steals active window focus or cursor focus from your current editor or document.
- **Conflict-Free Triggering & Full Control**: Toggle recording with a quick **Double-Tap of the Right Command key (`⌘`)** or a single click on the idle pill. Right-click the pill anytime to toggle Ollama cleanup, restart the process, or quit cleanly.
- **Native Background macOS App**: Runs as a background service (`LSUIElement`) with no Dock clutter and no Terminal window. Easily install to `/Applications/Spokn.app` with optional automatic launch at login.
- **Personalized Accuracy**: Hot-reloadable dictionary corrections via gitignored `corrections.json`, optional local **Ollama** LLM cleanup to eliminate filler words ("um", "uh") and smooth stutters, and smart trailing spacing to avoid mashed words during consecutive dictations.

---

## How the Dictation Pill Works

The floating pill morphs smoothly across 5 interactive states driven by a 220 ms spring animation:

| State | Canvas Size | Visual Behavior & Interaction |
| :--- | :--- | :--- |
| **Idle** | `36 × 6 px` | Minimal translucent resting bar with 35% white border stroke. Clicking starts recording. Right-clicking opens the menu. |
| **Listening** | `124 × 36 px` | Springs open on trigger. Shows **× Cancel** button, a resting 10-bar flat dot waveform (2 px), and a coral **Stop** button. |
| **Speaking** | `124 × 36 px` | Transitions automatically when speech volume exceeds threshold. Drives a 60 FPS center-outward animated vocal waveform. |
| **Collapsing** | `100 × 28 px` | Triggered when recording stops. Flattens waveform, shrinks controls into 4 px dots, and morphs into processing in 220 ms. |
| **Processing** | `90 × 26 px` | Minimal compact pill displaying 3 sequential pulsing dots while MLX Whisper transcribes and pastes text at your cursor. |

---

## Quick Start

### 1. Prerequisites

Install required audio capture and multimedia libraries via [Homebrew](https://brew.sh/):

```bash
brew install portaudio ffmpeg
```

### 2. Clone & Setup Virtual Environment

```bash
git clone https://github.com/shantodev/Spokn.git
cd Spokn
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Install macOS Application & LaunchAgent

Install Spokn directly into `/Applications/Spokn.app` and configure it to start automatically on login:

```bash
python install_app.py --autostart
```

*(To uninstall the autostart agent later: `python install_app.py --remove-autostart`)*

### 4. Grant macOS Permissions

Open **System Settings → Privacy & Security** and grant access:
- **Microphone**: Allow `Spokn` (and your terminal / `.venv/bin/python` if running via CLI).
- **Accessibility**: Allow `Spokn` (and your terminal / `.venv/bin/python` if running via CLI) so Spokn can detect global Right Command double-taps and paste transcribed text via `Cmd+V`.

### 5. Launch & Dictate

Launch Spokn from Spotlight (`Cmd + Space` → `Spokn`) or terminal:

```bash
open /Applications/Spokn.app
```

- **Double-tap Right Command (`cmd_r`)** or click the idle pill to speak.
- Double-tap again (or click the coral Stop button) to stop; Spokn transcribes and automatically pastes into your active cursor.
- Click **× Cancel** on the pill anytime to discard audio without transcribing.
- Right-click the pill to access options (**Ollama Cleanup**, **Restart Spokn**, **Quit Spokn**).
- Inspect logs anytime:
  ```bash
  tail -f ~/Library/Logs/Spokn.log
  ```

---

## Personalizing Accuracy

### 1. Custom Vocabulary & Hot-Reload Corrections

Whisper sometimes misinterprets niche proper nouns or technical jargon. Spokn supports whole-word regex replacements using a local, gitignored `corrections.json` file.

Copy the provided example template:

```bash
cp corrections.example.json corrections.json
```

Add your custom replacements (case-insensitive):

```json
{
  "pie side": "PySide6",
  "code forces": "Codeforces",
  "anti gravity": "Antigravity"
}
```

`corrections.json` is re-read on every transcription, allowing you to add or modify rules on the fly without restarting Spokn.

### 2. Optional Local Ollama Post-Processing

Spokn can optionally route transcribed text through a local LLM via [Ollama](https://ollama.com/) to remove verbal fillers ("um", "uh", "you know"), polish grammar, and fix punctuation before pasting:

1. Install and start Ollama, then pull the lightweight Llama 3.2 model:
   ```bash
   ollama pull llama3.2:3b
   ```
2. Toggle **Ollama Cleanup** by right-clicking the floating Dictation Pill (or set `USE_OLLAMA_CLEANUP = True` in `config.py`).
3. If Ollama is unreachable or takes longer than `1.5s`, Spokn automatically falls back to raw Whisper output instantly.

---

## Configuration

All customizable parameters are centrally defined in [`config.py`](file:///Users/shanto/Developer/Spokn/config.py):

| Parameter | Current Default | Description |
| :--- | :--- | :--- |
| `APP_NAME` | `"Spokn"` | Application display name |
| `APP_VERSION` | `"1.1.0"` | Current release version |
| `MODEL_REPO` | `"mlx-community/whisper-large-v3-turbo"` | High-accuracy multilingual model optimized for Apple Silicon unified memory |
| `LANGUAGE` | `"en"` | Language code pinned to prevent hallucinations or drift |
| `INITIAL_PROMPT` | *(Technical proper nouns)* | Vocabulary hints conditioning the tokenizer on domain terms |
| `HOTKEY` | `"cmd_r"` | Right Command key (`cmd_r`). Alternatives: `"f5"` (Mic key), `"alt_r"` |
| `HOTKEY_MODE` | `"double_tap"` | Triggering mode (`"double_tap"`, `"toggle"`, `"hold"`) |
| `DOUBLE_TAP_WINDOW` | `0.35` | Max interval (seconds) between two taps to register a trigger |
| `HOTKEY_DEBOUNCE_SECONDS` | `0.35` | Debounce window to filter physical key chatter |
| `SAMPLE_RATE` | `16000` | 16 kHz audio sample rate required by Whisper |
| `CHANNELS` | `1` | Mono audio capture |
| `DTYPE` | `"int16"` | 16-bit PCM integer recording format |
| `SPEECH_THRESHOLD` | `0.008` | RMS threshold to transition pill from Listening to Speaking |
| `MIN_RECORDING_DURATION` | `0.3` | Minimum seconds of audio required before transcribing |
| `CLIPBOARD_RESTORE_DELAY` | `0.15` | Seconds after `Cmd+V` simulation before restoring previous clipboard |
| `TRAILING_SPACE` | `True` | Appends a trailing space if text doesn't end in whitespace |
| `USE_OLLAMA_CLEANUP` | `False` | Optional local Ollama grammar/filler cleanup toggle |
| `OLLAMA_MODEL` | `"llama3.2:3b"` | Local Ollama model tag |
| `OLLAMA_URL` | `"http://localhost:11434/api/generate"` | Local Ollama endpoint |
| `OLLAMA_TIMEOUT` | `1.5` | Timeout (seconds) before falling back to raw Whisper text |

---

## Project Layout

Spokn follows a lean 4-file architecture:

```
Spokn/
├── config.py              # Central application configuration & thresholds
├── dictate.py             # Dictation coordinator, audio recorder & MLX transcriber
├── overlay.py             # Non-activating floating PySide6 Dictation Pill overlay
├── install_app.py         # macOS .app bundle & launch-at-login installer
├── corrections.json       # Personal word replacements (local, gitignored)
├── corrections.example.json # Generic example dictionary template
├── requirements.txt       # Python package dependencies
└── logo.png               # Spokn vector source logo & icon asset
```

---

## License

MIT License. Built for Apple Silicon.
