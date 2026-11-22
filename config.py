"""
Spokn - Central Configuration
All configurable parameters for audio recording, MLX Whisper transcription,
hotkey bindings, and clipboard injection are defined here.
"""

# Application identity
APP_NAME = "Spokn"

# Model repository on Hugging Face / MLX
# Default: Fast, lightweight English model with low memory footprint and latency.
# Optional upgrade: "mlx-community/whisper-large-v3-turbo" (multilingual, higher accuracy)
MODEL_REPO = "mlx-community/whisper-small.en-mlx"

# Global hotkey configuration
# Key to hold down to record and release to transcribe.
# "alt_r" represents the Right Option key on macOS keyboards.
HOTKEY = "alt_r"

# Audio capture settings
# Whisper requires 16kHz mono audio.
SAMPLE_RATE = 16000  # 16kHz sample rate
CHANNELS = 1         # Mono audio
DTYPE = "int16"      # Standard 16-bit PCM integer format

# Recording thresholds
# Minimum recording duration in seconds; ignore shorter keypresses to prevent accidental triggers.
MIN_RECORDING_DURATION = 0.3

# Clipboard automation timing
# Delay in seconds to wait after sending Cmd+V before restoring original clipboard contents.
CLIPBOARD_RESTORE_DELAY = 0.15
