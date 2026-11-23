#!/usr/bin/env python3
"""
Spokn - macOS Application & Background Service Installer

Creates a lightweight, native macOS application bundle at /Applications/Spokn.app
configured with LSUIElement=true so it runs completely in the background with:
- No Dock icon
- No Terminal window
- Persistent menu-bar status icon & floating Dictation Pill overlay
- Automated stdout/stderr logging to ~/Library/Logs/Spokn.log

Usage:
    python install_app.py                 # Install /Applications/Spokn.app
    python install_app.py --autostart     # Install and configure launch-at-login
    python install_app.py --remove-autostart  # Remove launch-at-login LaunchAgent
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import plistlib
import shutil
import subprocess
import sys


APP_NAME = "Spokn"
BUNDLE_ID = "com.spokn.app"
APP_PATH = Path("/Applications") / f"{APP_NAME}.app"
LAUNCH_AGENTS_DIR = Path.home() / "Library" / "LaunchAgents"
LAUNCH_AGENT_PLIST = LAUNCH_AGENTS_DIR / f"{BUNDLE_ID}.plist"
LOG_PATH = Path.home() / "Library" / "Logs" / f"{APP_NAME}.log"


def install_app_bundle(repo_dir: Path, target_path: Path = APP_PATH) -> Path:
    """Build and install Spokn.app bundle."""
    venv_python = repo_dir / ".venv" / "bin" / "python"
    entry_script = repo_dir / "dictate.py"

    if not venv_python.exists():
        print(f"[!] Virtual environment not found at {venv_python}")
        print("    Please create it first: python3 -m venv .venv && pip install -r requirements.txt")
        sys.exit(1)

    print(f"[*] Installing {target_path}...")

    # Remove previous installation if exists
    if target_path.exists():
        if target_path.is_dir():
            shutil.rmtree(target_path)
        else:
            target_path.unlink()

    contents_dir = target_path / "Contents"
    macos_dir = contents_dir / "MacOS"
    resources_dir = contents_dir / "Resources"
    macos_dir.mkdir(parents=True, exist_ok=True)
    resources_dir.mkdir(parents=True, exist_ok=True)

    # 1. Write Contents/Info.plist with LSUIElement=True (no Dock icon, silent agent)
    info_plist_data = {
        "CFBundleName": APP_NAME,
        "CFBundleDisplayName": APP_NAME,
        "CFBundleIdentifier": BUNDLE_ID,
        "CFBundleVersion": "1.0.0",
        "CFBundleShortVersionString": "1.0.0",
        "CFBundlePackageType": "APPL",
        "CFBundleExecutable": APP_NAME,
        "LSUIElement": True,
        "NSMicrophoneUsageDescription": (
            "Spokn needs microphone access for offline local speech-to-text dictation."
        ),
        "NSHighResolutionCapable": True,
    }

    plist_path = contents_dir / "Info.plist"
    with open(plist_path, "wb") as f:
        plistlib.dump(info_plist_data, f, fmt=plistlib.FMT_XML)

    # 2. Write Contents/MacOS/Spokn launcher script
    launcher_path = macos_dir / APP_NAME
    launcher_content = f"""#!/usr/bin/env bash
set -e

# Spokn Background macOS Application Launcher
PROJECT_DIR="{repo_dir.resolve()}"
PYTHON_BIN="$PROJECT_DIR/.venv/bin/python"
ENTRY_SCRIPT="$PROJECT_DIR/dictate.py"
LOG_FILE="$HOME/Library/Logs/{APP_NAME}.log"

mkdir -p "$HOME/Library/Logs"

cd "$PROJECT_DIR"
exec "$PYTHON_BIN" "$ENTRY_SCRIPT" >> "$LOG_FILE" 2>&1
"""

    with open(launcher_path, "w", encoding="utf-8") as f:
        f.write(launcher_content)

    os.chmod(launcher_path, 0o755)

    print(f"[+] Successfully installed {APP_NAME}.app to {target_path}")
    return target_path


def enable_autostart(app_path: Path = APP_PATH) -> None:
    """Create and load a macOS LaunchAgent to launch Spokn.app on user login."""
    print(f"[*] Configuring launch-at-login in {LAUNCH_AGENT_PLIST}...")

    LAUNCH_AGENTS_DIR.mkdir(parents=True, exist_ok=True)

    agent_data = {
        "Label": BUNDLE_ID,
        "ProgramArguments": ["/usr/bin/open", "-a", str(app_path.resolve())],
        "RunAtLoad": True,
        "KeepAlive": False,
    }

    with open(LAUNCH_AGENT_PLIST, "wb") as f:
        plistlib.dump(agent_data, f, fmt=plistlib.FMT_XML)

    # Load agent with launchctl
    subprocess.run(["launchctl", "load", str(LAUNCH_AGENT_PLIST)], check=False)
    print(f"[+] LaunchAgent registered. Spokn will now start automatically at login.")


def remove_autostart() -> None:
    """Unload and delete the launch-at-login LaunchAgent."""
    print(f"[*] Removing launch-at-login LaunchAgent...")
    if LAUNCH_AGENT_PLIST.exists():
        subprocess.run(["launchctl", "unload", str(LAUNCH_AGENT_PLIST)], check=False)
        LAUNCH_AGENT_PLIST.unlink()
        print(f"[+] Removed {LAUNCH_AGENT_PLIST}")
    else:
        print(f"[*] No existing LaunchAgent found at {LAUNCH_AGENT_PLIST}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Install Spokn as a native macOS background application.")
    parser.add_argument(
        "--target",
        type=Path,
        default=APP_PATH,
        help="Target installation path for the application bundle (default: /Applications/Spokn.app).",
    )
    parser.add_argument(
        "--autostart",
        action="store_true",
        help="Configure Spokn.app to launch automatically at login via LaunchAgent.",
    )
    parser.add_argument(
        "--remove-autostart",
        action="store_true",
        help="Remove the LaunchAgent so Spokn no longer starts automatically at login.",
    )

    args = parser.parse_args()

    repo_dir = Path(__file__).resolve().parent

    if args.remove_autostart:
        remove_autostart()
        return

    # Install the app bundle
    target_app = install_app_bundle(repo_dir, args.target)

    if args.autostart:
        enable_autostart(target_app)

    print("\n" + "=" * 65)
    print("Spokn Installation Complete!")
    print("=" * 65)
    print(f"App Location:      {target_app}")
    print(f"Log Output:        tail -f {LOG_PATH}")
    print("\nNext Steps:")
    print("1. Launch Spokn:")
    print(f"   open {target_app}")
    print("\n2. Grant macOS Permissions (required for dictation & paste):")
    print("   - System Settings -> Privacy & Security -> Microphone -> allow 'Spokn'")
    print("   - System Settings -> Privacy & Security -> Accessibility -> allow 'Spokn'")
    print("\n3. Dictate:")
    print("   - Double-tap Right Command (cmd_r) or click the pill to talk!")
    print("=" * 65)


if __name__ == "__main__":
    main()
