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


def generate_squircle_png(logo_path: Path, output_png_path: Path) -> bool:
    """Render a 1024x1024 macOS squircle icon with the bold white S symbol."""
    try:
        from PySide6.QtGui import QImage, QColor, QPainter, QPen, QBrush
        from PySide6.QtCore import Qt, QRectF
    except ImportError:
        # Fallback to invoking .venv python if running under system python without PySide6
        venv_python = logo_path.parent / ".venv" / "bin" / "python"
        if venv_python.exists():
            script = f"""
from PySide6.QtGui import QImage, QColor, QPainter, QPen, QBrush
from PySide6.QtCore import Qt, QRectF
img = QImage('{logo_path}')
min_x, max_x = img.width(), 0
min_y, max_y = img.height(), 0
for y in range(img.height()):
    for x in range(img.width()):
        c = QColor(img.pixel(x, y))
        lum = 0.299 * c.red() + 0.587 * c.green() + 0.114 * c.blue()
        if lum > 128:
            if x < min_x: min_x = x
            if x > max_x: max_x = x
            if y < min_y: min_y = y
            if y > max_y: max_y = y
crop_w = max_x - min_x + 1
crop_h = max_y - min_y + 1
s_img = QImage(crop_w, crop_h, QImage.Format.Format_ARGB32)
s_img.fill(Qt.GlobalColor.transparent)
for y in range(crop_h):
    for x in range(crop_w):
        c = QColor(img.pixel(min_x + x, min_y + y))
        lum = 0.299 * c.red() + 0.587 * c.green() + 0.114 * c.blue()
        if lum > 40:
            alpha = int(min(255, lum))
            s_img.setPixelColor(x, y, QColor(255, 255, 255, alpha))
canvas = QImage(1024, 1024, QImage.Format.Format_ARGB32)
canvas.fill(Qt.GlobalColor.transparent)
painter = QPainter(canvas)
painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
squircle_rect = QRectF(92.0, 92.0, 840.0, 840.0)
pen = QPen(QColor(255, 255, 255, 31), 2.0)
painter.setPen(pen)
painter.setBrush(QBrush(QColor('#0A0A0A')))
painter.drawRoundedRect(squircle_rect, 185.0, 185.0)
s_rect = QRectF(252.0, 252.0, 520.0, 520.0)
painter.drawImage(s_rect, s_img)
painter.end()
canvas.save('{output_png_path}')
"""
            res = subprocess.run([str(venv_python), "-c", script], capture_output=True)
            return res.returncode == 0
        return False

    img = QImage(str(logo_path))
    if img.isNull():
        return False

    min_x, max_x = img.width(), 0
    min_y, max_y = img.height(), 0

    for y in range(img.height()):
        for x in range(img.width()):
            c = QColor(img.pixel(x, y))
            lum = 0.299 * c.red() + 0.587 * c.green() + 0.114 * c.blue()
            if lum > 128:
                if x < min_x: min_x = x
                if x > max_x: max_x = x
                if y < min_y: min_y = y
                if y > max_y: max_y = y

    if min_x > max_x or min_y > max_y:
        min_x, max_x, min_y, max_y = 0, img.width() - 1, 0, img.height() - 1

    crop_w = max_x - min_x + 1
    crop_h = max_y - min_y + 1

    s_img = QImage(crop_w, crop_h, QImage.Format.Format_ARGB32)
    s_img.fill(Qt.GlobalColor.transparent)

    for y in range(crop_h):
        for x in range(crop_w):
            c = QColor(img.pixel(min_x + x, min_y + y))
            lum = 0.299 * c.red() + 0.587 * c.green() + 0.114 * c.blue()
            if lum > 40:
                alpha = int(min(255, lum))
                s_img.setPixelColor(x, y, QColor(255, 255, 255, alpha))

    canvas = QImage(1024, 1024, QImage.Format.Format_ARGB32)
    canvas.fill(Qt.GlobalColor.transparent)

    painter = QPainter(canvas)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

    # Squircle 840x840 centered, 185 px radius, #0A0A0A fill, 2px rgba(255,255,255,0.12) border
    squircle_rect = QRectF(92.0, 92.0, 840.0, 840.0)
    pen = QPen(QColor(255, 255, 255, 31), 2.0)
    painter.setPen(pen)
    painter.setBrush(QBrush(QColor("#0A0A0A")))
    painter.drawRoundedRect(squircle_rect, 185.0, 185.0)

    # S symbol centered, ~520 px tall (62% of squircle)
    s_rect = QRectF(252.0, 252.0, 520.0, 520.0)
    painter.drawImage(s_rect, s_img)
    painter.end()

    return canvas.save(str(output_png_path))


def generate_icns(logo_path: Path, output_icns_path: Path) -> bool:
    """Convert logo.png into a native macOS .icns file using PySide6 and iconutil."""
    if not logo_path.exists():
        print(f"[!] logo.png not found at {logo_path}, skipping .icns generation.")
        return False

    temp_dir = Path("/tmp") / "spokn_icon_build"
    iconset_dir = temp_dir / "Spokn.iconset"
    master_png = temp_dir / "Spokn_1024.png"

    try:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        iconset_dir.mkdir(parents=True, exist_ok=True)

        # 1. Render 1024x1024 macOS squircle PNG with bold white S
        if not generate_squircle_png(logo_path, master_png):
            print("[!] Failed to render master squircle PNG.")
            return False

        # 2. Standard macOS iconset dimensions
        sizes = [
            ("icon_16x16.png", 16),
            ("icon_16x16@2x.png", 32),
            ("icon_32x32.png", 32),
            ("icon_32x32@2x.png", 64),
            ("icon_128x128.png", 128),
            ("icon_128x128@2x.png", 256),
            ("icon_256x256.png", 256),
            ("icon_256x256@2x.png", 512),
            ("icon_512x512.png", 512),
            ("icon_512x512@2x.png", 1024),
        ]

        for filename, size in sizes:
            out_file = iconset_dir / filename
            subprocess.run(
                [
                    "sips",
                    "-z",
                    str(size),
                    str(size),
                    str(master_png),
                    "--out",
                    str(out_file),
                ],
                check=True,
                capture_output=True,
            )

        # 3. Convert iconset to icns using iconutil
        output_icns_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["iconutil", "-c", "icns", str(iconset_dir), "-o", str(output_icns_path)],
            check=True,
            capture_output=True,
        )
        print(f"[+] Generated bold application icon at {output_icns_path}")
        return True
    except Exception as exc:
        print(f"[!] Warning generating .icns: {exc}")
        return False
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


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

    # Generate Spokn.icns icon
    generate_icns(repo_dir / "logo.png", resources_dir / "Spokn.icns")

    # 1. Write Contents/Info.plist with LSUIElement=True (no Dock icon, silent agent)
    info_plist_data = {
        "CFBundleName": APP_NAME,
        "CFBundleDisplayName": APP_NAME,
        "CFBundleIdentifier": BUNDLE_ID,
        "CFBundleVersion": "1.1.0",
        "CFBundleShortVersionString": "1.1.0",
        "CFBundleIconFile": "Spokn",
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

    # Touch bundle to notify Finder / Spotlight of metadata and icon update
    subprocess.run(["touch", str(target_path)], check=False)

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
