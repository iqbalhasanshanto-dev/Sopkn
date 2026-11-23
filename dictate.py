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

import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time
from typing import Any
import urllib.request
import AppKit
import mlx_whisper
import numpy as np
from pynput import keyboard
from PySide6.QtCore import QCoreApplication, QObject, QPoint, QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QBrush, QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QApplication, QMenu
import sounddevice as sd

import config
from overlay import DictationPillOverlay, PillState


LOCK_FILE_PATH = "/tmp/spokn.lock"
_lock_file_fd: int | None = None


def acquire_single_instance_lock() -> int:
    """Acquire a non-blocking exclusive file lock on /tmp/spokn.lock.

    If another instance of Spokn is already running, exits cleanly with sys.exit(0)
    so launching Spokn while it is already active never spawns a conflicting duplicate.
    """
    global _lock_file_fd
    try:
        fd = os.open(LOCK_FILE_PATH, os.O_CREAT | os.O_RDWR, 0o600)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _lock_file_fd = fd
        return fd
    except (BlockingIOError, OSError):
        print(f"[{config.APP_NAME}] Already running. Exiting duplicate instance.", flush=True)
        sys.exit(0)


def release_single_instance_lock() -> None:
    """Release the single-instance file lock if held."""
    global _lock_file_fd
    if _lock_file_fd is not None:
        try:
            fcntl.flock(_lock_file_fd, fcntl.LOCK_UN)
            os.close(_lock_file_fd)
        except Exception:
            pass
        _lock_file_fd = None


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


def clean_with_ollama(text: str) -> str:
    """Clean transcribed text using a local Ollama model (fix punctuation, remove filler words).

    Falls back immediately and silently to original text if Ollama is unreachable or times out.
    """
    if not text or not text.strip():
        return text

    system_instruction = (
        "You are an expert speech-to-text post-processor. "
        "Clean up the following dictated text by fixing punctuation, capitalization, and grammar, "
        "removing verbal filler words (e.g. 'um', 'uh', 'you know', 'like'), and smoothing stutters. "
        "Preserve the exact original meaning, vocabulary, and technical terms. "
        "Output ONLY the cleaned transcription with no surrounding quotes, markdown formatting, or preamble."
    )

    payload = json.dumps({
        "model": getattr(config, "OLLAMA_MODEL", "llama3.2:3b"),
        "prompt": text,
        "system": system_instruction,
        "stream": False,
        "options": {
            "temperature": 0.1,
        },
    }).encode("utf-8")

    req = urllib.request.Request(
        getattr(config, "OLLAMA_URL", "http://localhost:11434/api/generate"),
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    try:
        timeout_sec = getattr(config, "OLLAMA_TIMEOUT", 3.0)
        with urllib.request.urlopen(req, timeout=timeout_sec) as response:
            if response.status == 200:
                body = json.loads(response.read().decode("utf-8"))
                cleaned = body.get("response", "").strip()
                if cleaned:
                    if (cleaned.startswith('"') and cleaned.endswith('"')) or (
                        cleaned.startswith("'") and cleaned.endswith("'")
                    ):
                        cleaned = cleaned[1:-1].strip()
                    return cleaned
    except Exception as exc:
        print(f"[{config.APP_NAME}] Ollama cleanup bypassed ({exc}). Using Whisper output.", flush=True)

    return text


# 5 speech-range frequency bands spanning ~120 Hz to 3800 Hz
# (from strongest vocal fundamentals b0 to high presence b4)
SPEECH_BAND_EDGES = [120.0, 280.0, 650.0, 1400.0, 2600.0, 3800.0]


class AudioRecorder:
    """Handles real-time microphone audio capture and normalization using sounddevice.

    Thread Safety:
    - Never calls Qt functions or emits Qt signals from _audio_callback (the PortAudio
      real-time audio thread).
    - Computes peak-hold RMS volume and a 5-band speech spectrum, storing them
      atomically for the Qt main thread to poll safely via consume_audio_levels().
    """

    def __init__(self) -> None:
        """Initializes internal audio chunk buffer, recording flag, and stream handle."""
        self._buffer: list[np.ndarray] = []
        self._is_recording: bool = False
        self._stream: sd.InputStream | None = None
        self._peak_rms: float = 0.0
        self._peak_bands: list[float] = [0.0] * 5
        self._stream_lock = threading.Lock()

    @property
    def is_recording(self) -> bool:
        """Return whether audio is currently actively recording."""
        return self._is_recording

    @property
    def latest_rms(self) -> float:
        """Return the current peak RMS volume since the last consumption."""
        return self._peak_rms

    def consume_audio_levels(self) -> tuple[float, list[float]]:
        """Atomically read and reset peak RMS and 5-band spectrum since the last poll."""
        rms = self._peak_rms
        bands = list(self._peak_bands)
        self._peak_rms = 0.0
        self._peak_bands = [0.0] * 5
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

            # Compute 5-band speech spectrum (~120 Hz to 3800 Hz) via rfft
            rfft_mag = np.abs(np.fft.rfft(data_float))
            freqs = np.fft.rfftfreq(n_samples, d=1.0 / config.SAMPLE_RATE)
            fft_norm = rfft_mag / (n_samples / 2.0)

            for i in range(5):
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
        with self._stream_lock:
            if self._is_recording:
                return True

            self._buffer.clear()
            self._peak_rms = 0.0
            self._peak_bands = [0.0] * 5
            try:
                self._stream = sd.InputStream(
                    samplerate=config.SAMPLE_RATE,
                    channels=config.CHANNELS,
                    dtype=config.DTYPE,
                    blocksize=1024,
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
        with self._stream_lock:
            self._is_recording = False
            self._peak_rms = 0.0
            self._peak_bands = [0.0] * 5

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
        with self._stream_lock:
            self._is_recording = False
            self._peak_rms = 0.0
            self._peak_bands = [0.0] * 5
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
                condition_on_previous_text=False,
            )
            if isinstance(result, dict):
                text_val = result.get("text", "")
                if isinstance(text_val, str):
                    return text_val.strip()
            return ""
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
        self.overlay.idle_clicked.connect(self._on_pill_idle_clicked)
        self.overlay.cancel_clicked.connect(self._on_pill_cancel_clicked)
        self.overlay.stop_clicked.connect(self._on_pill_stop_clicked)

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
        self._hotkey_is_down: bool = False

        # Last active third-party application tracking (for focus restoration)
        self._last_active_app: Any = None
        self._record_last_active_app()

        # Polling timer to keep last active app updated every 300 ms
        self._app_tracker_timer = QTimer(self.overlay)
        self._app_tracker_timer.setInterval(300)
        self._app_tracker_timer.timeout.connect(self._record_last_active_app)
        self._app_tracker_timer.start()

        # Context menu on right-clicking the floating Dictation Pill
        self.use_ollama_cleanup: bool = getattr(config, "USE_OLLAMA_CLEANUP", False)
        self._pill_menu = QMenu()
        self._pill_status_action = self._pill_menu.addAction(
            f"{config.APP_NAME} v{config.APP_VERSION} — Ready"
        )
        self._pill_status_action.setEnabled(False)

        self._pill_toggle_action = self._pill_menu.addAction("Start / Stop Dictation")
        self._pill_toggle_action.triggered.connect(self.handle_hotkey_toggle)

        self._pill_ollama_action = self._pill_menu.addAction("Ollama Cleanup")
        self._pill_ollama_action.setCheckable(True)
        self._pill_ollama_action.setChecked(self.use_ollama_cleanup)
        self._pill_ollama_action.toggled.connect(self._set_ollama_cleanup)

        self._pill_menu.addSeparator()

        self._pill_restart_action = self._pill_menu.addAction("Restart Spokn")
        self._pill_restart_action.triggered.connect(lambda: self.restart_and_reexec(QApplication.instance()))

        self._pill_quit_action = self._pill_menu.addAction("Quit Spokn")
        self._pill_quit_action.triggered.connect(lambda: self.stop_and_exit(QApplication.instance()))

        self.overlay.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.overlay.customContextMenuRequested.connect(self._on_overlay_context_menu)

    def _record_last_active_app(self) -> None:
        """Track the most recent frontmost application that is not Spokn/Python."""
        try:
            import os
            import AppKit

            workspace = AppKit.NSWorkspace.sharedWorkspace()  # type: ignore[attr-defined]
            front_app = workspace.frontmostApplication()
            if front_app is not None and front_app.processIdentifier() != os.getpid():
                self._last_active_app = front_app
        except Exception:
            pass

    def _restore_last_active_app(self) -> None:
        """Re-activate the last active non-Spokn app so cursor focus is restored."""
        try:
            import os
            import AppKit

            if self._last_active_app is None:
                self._record_last_active_app()

            if self._last_active_app is None:
                return

            # NSApplicationActivateIgnoringOtherApps = 1 << 1
            self._last_active_app.activateWithOptions_(1 << 1)
            time.sleep(0.08)
        except Exception:
            pass

    def _on_pill_idle_clicked(self) -> None:
        """Pill clicked while idle -> restore target app focus and start recording."""
        self._restore_last_active_app()
        self.start_dictation()

    def _on_pill_stop_clicked(self) -> None:
        """Stop button clicked on pill -> restore target app focus and stop recording."""
        self._restore_last_active_app()
        self.stop_dictation()

    def _on_pill_cancel_clicked(self) -> None:
        """Cancel button clicked on pill -> restore target app focus and cancel recording."""
        self._restore_last_active_app()
        self.cancel_dictation()

    def _set_ollama_cleanup(self, enabled: bool) -> None:
        """Toggle Ollama post-processing cleanup live."""
        self.use_ollama_cleanup = enabled
        if hasattr(self, "_pill_ollama_action") and self._pill_ollama_action is not None:
            self._pill_ollama_action.setChecked(enabled)
        status = "enabled" if enabled else "disabled"
        print(f"[{config.APP_NAME}] Ollama Cleanup {status}.", flush=True)

    def _on_overlay_context_menu(self, pos: QPoint) -> None:
        """Display context menu when right-clicking on the pill."""
        pill_rect = QRectF(
            self.overlay.PADDING,
            self.overlay.PADDING,
            self.overlay._current_w,
            self.overlay._current_h,
        )
        if pill_rect.contains(pos):
            self._pill_menu.exec(self.overlay.mapToGlobal(pos))

    def _update_tray_status(self, status: str) -> None:
        """Update live status in the pill context menu."""
        if hasattr(self, "_pill_status_action") and self._pill_status_action is not None:
            self._pill_status_action.setText(f"{config.APP_NAME} v{config.APP_VERSION} — {status}")

    def start(self) -> None:
        """Show overlay, start global keyboard listener, and warm up model in background."""
        self.overlay.show()

        # Check accessibility permissions
        try:
            import ApplicationServices
            if not ApplicationServices.AXIsProcessTrusted():  # type: ignore[attr-defined]
                print(
                    f"[{config.APP_NAME}] Warning: macOS Accessibility permissions are not yet enabled for Spokn.\n"
                    f"    Please grant access in System Settings -> Privacy & Security -> Accessibility\n"
                    f"    so Spokn can monitor Right Command double-taps and paste text automatically.",
                    flush=True,
                )
        except Exception:
            pass

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
                f"[{config.APP_NAME} v{config.APP_VERSION}] Ready. Double-tap Right Command or click pill to talk (Ctrl+C to quit).",
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
            self._update_tray_status("Listening")
            self._level_timer.start()
            print(f"[{config.APP_NAME}] Listening...", flush=True)

    def stop_dictation(self) -> None:
        """Stop recording, trigger COLLAPSING -> PROCESSING immediately, and transcribe on worker thread."""
        with self._lock:
            if not self.recorder.is_recording or self.is_processing:
                return
            self.is_processing = True

        # Immediately update UI state on Qt main thread without waiting for PortAudio
        self._level_timer.stop()
        self.overlay.set_audio_bands([0.0] * 10)
        self.overlay.set_audio_level(0.0)

        # Transition to COLLAPSING (which auto-morphs to PROCESSING after 220 ms)
        self._current_state = PillState.PROCESSING
        self.overlay.set_state(PillState.COLLAPSING)
        self._update_tray_status("Transcribing")
        print(f"[{config.APP_NAME}] Transcribing...", flush=True)

        def _stop_and_process_worker() -> None:
            # Stop audio stream in background worker thread (never blocks Qt event loop)
            audio = self.recorder.stop()
            if audio is None:
                with self._lock:
                    self.is_processing = False
                    self._current_state = PillState.IDLE
                self._update_tray_status("Ready")
                try:
                    self.ui_bridge.state_changed.emit(PillState.IDLE)
                except RuntimeError:
                    pass
                print(f"[{config.APP_NAME}] Ignored short recording.", flush=True)
                print(f"[{config.APP_NAME}] Ready.", flush=True)
                return

            self.process_audio(audio)

        worker = threading.Thread(
            target=_stop_and_process_worker,
            daemon=True,
            name="AudioProcessorThread",
        )
        worker.start()

    def cancel_dictation(self) -> None:
        """Cancel recording immediately and return to IDLE without transcribing."""
        with self._lock:
            if not self.recorder.is_recording:
                return

        # Immediately reset UI state on Qt main thread
        self._level_timer.stop()
        self.overlay.set_audio_bands([0.0] * 10)
        self.overlay.set_audio_level(0.0)
        self._current_state = PillState.IDLE
        self.overlay.set_state(PillState.IDLE)
        self._update_tray_status("Ready")
        print(f"[{config.APP_NAME}] Dictation canceled.", flush=True)
        print(f"[{config.APP_NAME}] Ready.", flush=True)

        # Stop and close audio stream asynchronously on worker thread
        threading.Thread(
            target=self.recorder.cancel,
            daemon=True,
            name="AudioCancelThread",
        ).start()

    def _poll_audio_level(self) -> None:
        """Polled on Qt main thread every 33 ms while recording to update UI state and vocal waveform."""
        if not self.recorder.is_recording:
            return

        rms, raw_bands = self.recorder.consume_audio_levels()

        # Transition from LISTENING to SPEAKING once microphone volume exceeds threshold
        if self._current_state == PillState.LISTENING and rms >= config.SPEECH_THRESHOLD:
            self._current_state = PillState.SPEAKING
            self.overlay.set_state(PillState.SPEAKING)

        # Modulate live 60 FPS waveform with calibrated MacBook Air mic sensitivity
        if self._current_state in (PillState.LISTENING, PillState.SPEAKING):
            NOISE_FLOOR = 0.0015
            PEAK_SPEECH = 0.024

            if rms <= NOISE_FLOOR:
                speech_level = 0.0
            else:
                raw_norm = min(1.0, max(0.0, (rms - NOISE_FLOOR) / (PEAK_SPEECH - NOISE_FLOOR)))
                speech_level = raw_norm ** 0.55

            self.overlay.set_audio_level(speech_level)

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
                if self.use_ollama_cleanup:
                    text = clean_with_ollama(text)

                if getattr(config, "TRAILING_SPACE", True) and text and not text[-1].isspace():
                    text += " "

                print(f"[{config.APP_NAME}] Transcribed: {text}", flush=True)
                self._restore_last_active_app()
                paste_text(text)
        finally:
            with self._lock:
                self.is_processing = False
                self._current_state = PillState.IDLE
            self._update_tray_status("Ready")
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
        if self._hotkey_is_down:
            # Ignore macOS modifier key-repeat events so _last_tap_time is never reset during hold
            return
        self._hotkey_is_down = True
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

        self._hotkey_is_down = False

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
    # Shutdown & Restart
    # -------------------------------------------------------------------------

    def restart_and_reexec(self, app: QCoreApplication | None) -> None:
        """Cleanly shut down Spokn, release the instance lock, and re-exec the process."""
        print(f"\n[{config.APP_NAME}] Restarting...", flush=True)
        try:
            self.stop_and_exit(app, do_sys_exit=False)
        except Exception:
            pass
        release_single_instance_lock()
        try:
            os.execv(sys.executable, [sys.executable, *sys.argv])
        except Exception as exc:
            print(f"[{config.APP_NAME}] Restart failed: {exc}", flush=True)
            os._exit(1)

    def stop_and_exit(self, app: QCoreApplication | None, do_sys_exit: bool = True) -> None:
        """Clean up audio, listener, overlay, and lock, then exit the application."""
        print(f"\n[{config.APP_NAME}] Exiting cleanly...", flush=True)
        try:
            self._level_timer.stop()
        except Exception:
            pass
        try:
            self._app_tracker_timer.stop()
        except Exception:
            pass
        try:
            self.overlay.set_audio_bands([0.0] * 10)
        except Exception:
            pass
        if self.recorder.is_recording:
            try:
                self.recorder.cancel()
            except Exception:
                pass
        if self._listener is not None and self._listener.is_alive():
            try:
                self._listener.stop()
            except Exception:
                pass
        try:
            self.overlay.close()
        except Exception:
            pass
        release_single_instance_lock()

        if do_sys_exit:
            if app is not None:
                try:
                    app.quit()
                except Exception:
                    pass
            # Immediate Force Quit fallback so background threads or PortAudio handles never hang
            os._exit(0)


def main() -> None:
    """Entry point for Spokn."""
    # Ensure only a single instance of Spokn runs at any time
    acquire_single_instance_lock()

    app = QApplication.instance() or QApplication(sys.argv)

    # Set macOS accessory activation policy immediately before constructing any UI elements
    try:
        import AppKit
        activation_policy_accessory = getattr(AppKit, "NSApplicationActivationPolicyAccessory", 1)
        AppKit.NSApp.setActivationPolicy_(activation_policy_accessory)  # type: ignore[attr-defined]
    except Exception:
        pass

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
