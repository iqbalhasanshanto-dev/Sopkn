"""
Spokn - Central Configuration
All configurable parameters for audio recording, MLX Whisper transcription,
hotkey bindings, visual overlay, and clipboard injection are defined here.
"""

# Application identity
APP_NAME = "Spokn"

# Model repository on Hugging Face / MLX
# Default: High-accuracy multilingual model optimized for Apple Silicon unified memory.
MODEL_REPO = "mlx-community/whisper-large-v3-turbo"
# Optional lighter/faster fallback: "mlx-community/whisper-small.en-mlx"

# Language and prompt context
# Lock Whisper to English to prevent hallucinations or language drifting on regional accents
LANGUAGE = "en"

# Custom vocabulary hint to condition Whisper's tokenizer on domain-specific proper nouns
INITIAL_PROMPT = (
    "Spokn, Gemini, Shanto, VUPC, Codeforces, ESP32, PySide6, Antigravity, "
    "macOS, Rajshahi, Computer Science and Engineering, astrophotography"
)

# Global hotkey configuration
# Default: "cmd_r" represents the Right Command key on macOS.
# Alternative: "f5" (Microphone key) or "alt_r" (Right Option).
HOTKEY = "cmd_r"

# Hotkey triggering behavior: "double_tap" (two quick taps of HOTKEY), "toggle", or "hold"
HOTKEY_MODE = "double_tap"

# Maximum time window in seconds between two taps for double-tap detection
DOUBLE_TAP_WINDOW = 0.35

# Debounce delay in seconds to ignore physical key-repeat / chatter
HOTKEY_DEBOUNCE_SECONDS = 0.35

# Audio capture settings
# Whisper requires 16kHz mono audio.
SAMPLE_RATE = 16000  # 16kHz sample rate
CHANNELS = 1         # Mono audio
DTYPE = "int16"      # Standard 16-bit PCM integer format

# Speech detection threshold
# Normalized RMS volume threshold (0.0 to 1.0) to transition the pill from LISTENING to SPEAKING
# Tuned for MacBook Air built-in microphone at desk distance (whisper ~0.005, desk speech ~0.015-0.035)
SPEECH_THRESHOLD = 0.008

# Recording thresholds
# Minimum recording duration in seconds; ignore shorter keypresses to prevent accidental triggers.
MIN_RECORDING_DURATION = 0.3

# Clipboard automation timing
# Delay in seconds to wait after sending Cmd+V before restoring original clipboard contents.
CLIPBOARD_RESTORE_DELAY = 0.15
