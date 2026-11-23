"""
Spokn - Dictation Pill Overlay (PySide6)

Exact implementation of the macOS Dictation Pill design from Figma:
- Non-activating, always-on-top, translucent desktop capsule overlay.
- Anchored bottom-center of the primary display with a fixed 12 px bottom inset.
- Five sequential morph states (Idle, Listening, Speaking, Collapsing, Processing)
  with a 220 ms spring animation.
- Zero text, no microphone icon, no glow, no gradient.
"""

from __future__ import annotations

from enum import Enum
import math
import signal
import sys
from typing import List

from PySide6.QtCore import (
    QEasingCurve,
    QPointF,
    QRectF,
    QSizeF,
    Qt,
    QTimer,
    QVariantAnimation,
    Signal,
)
from PySide6.QtGui import (
    QBrush,
    QColor,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPen,
)
from PySide6.QtWidgets import QApplication, QWidget


class PillState(str, Enum):
    """The 5 interactive visual states defined in the Figma specification."""
    IDLE = "idle"              # 36 × 6 px, resting bar, no shadow, 35% white border
    LISTENING = "listening"    # 124 × 36 px, Cancel + flat 10-dot waveform + Stop
    SPEAKING = "speaking"      # 124 × 36 px, Cancel + animated 10-bar waveform + Stop
    COLLAPSING = "collapsing"  # 100 × 28 px, flat waveform, controls shrink to 4 px dots
    PROCESSING = "processing"  # 90 × 26 px, flat waveform + 3 sequential pulsing dots


class DictationPillOverlay(QWidget):
    """Floating, non-activating macOS desktop overlay rendering the Spokn Dictation Pill.

    Window Behavior:
    - Uses Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint | Qt.Tool | Qt.WindowDoesNotAcceptFocus
      | Qt.NoDropShadowWindowHint and Qt.WA_ShowWithoutActivating so the overlay never steals
      input focus or cursor active state from the target application where the user is typing,
      and macOS does not draw an unwanted default window drop-shadow box.
    - Uses Qt.WA_TranslucentBackground with custom QPainter rendering so the capsule,
      border stroke, and soft elevation shadow blend natively over macOS desktop wallpaper.
    """

    # Interactive signals
    idle_clicked = Signal()
    cancel_clicked = Signal()
    stop_clicked = Signal()

    # Figma Geometry Tokens
    PADDING = 28.0            # Canvas padding so soft drop shadow never clips
    BOTTOM_INSET = 12.0       # Distance from bottom of screen to bottom edge of pill

    # Exact state dimensions (width, height)
    STATE_SIZES = {
        PillState.IDLE: (36.0, 6.0),
        PillState.LISTENING: (124.0, 36.0),
        PillState.SPEAKING: (124.0, 36.0),
        PillState.COLLAPSING: (100.0, 28.0),
        PillState.PROCESSING: (90.0, 26.0),
    }

    # Figma Colors
    COLOR_PILL_BG = QColor("#0A0A0A")          # Solid obsidian black
    COLOR_WHITE = QColor("#FFFFFF")            # Solid white
    COLOR_STOP_RED = QColor("#F0605D")         # Coral red for stop button
    COLOR_CANCEL_BG = QColor(255, 255, 255, 46) # Semi-transparent grey face (~18% white)
    COLOR_DOT_INACTIVE = QColor(255, 255, 255, 76) # Inactive dot grey (~30% white)

    # Figma Waveform Layout Constants (10 vertical bars, 2 px width, 3 px gap, 1 px radius)
    WAVEFORM_FLAT = [2.0] * 10

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        # macOS Focus-Safe, Shadow-Safe & Translucent Window Setup
        # Note: Do not use Qt.Tool as macOS automatically hides Qt.Tool windows when
        # another app is active. Use Qt.Window with native NSWindow configuration instead.
        self.setWindowFlags(
            Qt.Window
            | Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
            | Qt.WindowDoesNotAcceptFocus
            | Qt.NoDropShadowWindowHint
        )
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)

        # Current state and animated dimensions
        self._state = PillState.IDLE
        self._current_w = self.STATE_SIZES[PillState.IDLE][0]
        self._current_h = self.STATE_SIZES[PillState.IDLE][1]

        # Live vocal frequency bands (10 speech bands in range 0.0 to 1.0) and bar heights
        self._audio_bands: list[float] = [0.0] * 10
        self._bar_heights: list[float] = [2.0] * 10

        # Spring morph animation (220 ms OutBack)
        self._morph_anim = QVariantAnimation(self)
        self._morph_anim.setDuration(220)
        spring_curve = QEasingCurve(QEasingCurve.OutBack)
        spring_curve.setOvershoot(1.15)
        self._morph_anim.setEasingCurve(spring_curve)
        self._morph_anim.valueChanged.connect(self._on_morph_frame)

        # ~30 FPS (33 ms) waveform animation timer for live vocal amplitude
        self._waveform_timer = QTimer(self)
        self._waveform_timer.setInterval(33)
        self._waveform_timer.timeout.connect(self._on_waveform_tick)

        # 160 ms pulsing sequencer for Processing dots
        self._processing_phase = 0
        self._processing_timer = QTimer(self)
        self._processing_timer.setInterval(160)
        self._processing_timer.timeout.connect(self._on_processing_tick)

        # 220 ms auto-collapse timer (Collapsing -> Processing)
        self._collapse_timer = QTimer(self)
        self._collapse_timer.setSingleShot(True)
        self._collapse_timer.setInterval(220)
        self._collapse_timer.timeout.connect(lambda: self.set_state(PillState.PROCESSING))

        self._update_geometry()

    # -------------------------------------------------------------------------
    # Public State API
    # -------------------------------------------------------------------------

    @property
    def state(self) -> PillState:
        """Return the current visual state of the pill."""
        return self._state

    def set_state(self, state: PillState) -> None:
        """Transition the pill to a new visual state with a 220 ms spring animation."""
        if self._state == state:
            return

        self._state = state
        target_w, target_h = self.STATE_SIZES[state]

        # Manage animation timers based on new state
        if state in (PillState.LISTENING, PillState.SPEAKING):
            if not self._waveform_timer.isActive():
                self._waveform_timer.start()
        else:
            self._waveform_timer.stop()
            self._audio_bands = [0.0] * 10
            self._bar_heights = [2.0] * 10

        if state == PillState.PROCESSING:
            if not self._processing_timer.isActive():
                self._processing_timer.start()
        else:
            self._processing_timer.stop()

        if state == PillState.COLLAPSING:
            self._collapse_timer.start()
        else:
            self._collapse_timer.stop()

        # Start 220 ms spring morph animation
        self._morph_anim.stop()
        self._morph_anim.setStartValue(QSizeF(self._current_w, self._current_h))
        self._morph_anim.setEndValue(QSizeF(target_w, target_h))
        self._morph_anim.start()

    def set_audio_bands(self, bands: list[float]) -> None:
        """Set 10 live vocal frequency band amplitudes (each 0.0 to 1.0)."""
        if len(bands) == 10:
            self._audio_bands = [max(0.0, min(1.0, float(b))) for b in bands]

    def set_audio_level(self, level: float) -> None:
        """Set uniform audio level across all bands (fallback helper)."""
        clamped = max(0.0, min(1.0, float(level)))
        self._audio_bands = [clamped] * 10

    # -------------------------------------------------------------------------
    # Geometry & Animation Callbacks
    # -------------------------------------------------------------------------

    def _on_morph_frame(self, value: QSizeF) -> None:
        """Callback on each spring animation frame to update size and position."""
        self._current_w = value.width()
        self._current_h = value.height()
        self._update_geometry()
        self.update()

    def _on_waveform_tick(self) -> None:
        """Advance the live vocal waveform at ~30 FPS (33 ms) with fast attack and smooth decay."""
        for i in range(10):
            band = self._audio_bands[i] if i < len(self._audio_bands) else 0.0
            target_h = 2.0 + 16.0 * band

            current_h = self._bar_heights[i]
            if target_h > current_h:
                # Fast attack (0.55)
                current_h += (target_h - current_h) * 0.55
            else:
                # Smooth decay (0.22)
                current_h += (target_h - current_h) * 0.22

            # Settle flat to 2.0 px dots during pauses to prevent sub-pixel fractions
            if current_h < 2.05:
                current_h = 2.0

            self._bar_heights[i] = max(2.0, min(18.0, current_h))

        self.update()

    def _on_processing_tick(self) -> None:
        """Advance the sequential pulsing dot phase in Processing state."""
        self._processing_phase = (self._processing_phase + 1) % 3
        self.update()

    def _update_geometry(self) -> None:
        """Position overlay at horizontal center, 12 px above bottom of primary screen."""
        screen = QApplication.primaryScreen()
        if screen is None:
            return
        geo = screen.geometry()

        win_w = int(self._current_w + 2 * self.PADDING)
        win_h = int(self._current_h + 2 * self.PADDING)

        # Center horizontally on primary screen
        x = int(geo.x() + (geo.width() - win_w) / 2)
        # Anchor bottom of pill at exactly 12 px above primary screen bottom
        # Window coordinate equation: y + PADDING + current_h = geo.y() + geo.height() - BOTTOM_INSET
        y = int(geo.y() + geo.height() - self.BOTTOM_INSET - self.PADDING - self._current_h)

        self.setGeometry(x, y, win_w, win_h)

    # -------------------------------------------------------------------------
    # macOS Window Configuration
    # -------------------------------------------------------------------------

    def showEvent(self, event) -> None:
        """Configure native macOS NSWindow attributes when shown."""
        super().showEvent(event)
        self._configure_macos_window()

    def _configure_macos_window(self) -> None:
        """Configure native macOS NSWindow so the pill stays visible across all Spaces and apps."""
        try:
            import AppKit
            import objc

            NSFloatingWindowLevel = getattr(AppKit, "NSFloatingWindowLevel", 3)
            CAN_JOIN_ALL_SPACES = getattr(AppKit, "NSWindowCollectionBehaviorCanJoinAllSpaces", 1 << 0)
            STATIONARY = getattr(AppKit, "NSWindowCollectionBehaviorStationary", 1 << 4)
            FULL_SCREEN_AUX = getattr(AppKit, "NSWindowCollectionBehaviorFullScreenAuxiliary", 1 << 8)

            ns_view = objc.objc_object(c_void_p=int(self.winId()))
            ns_window = ns_view.window() if ns_view is not None else None
            if ns_window is not None:
                # Do not hide when Python/Spokn loses active application focus
                ns_window.setHidesOnDeactivate_(False)
                # Float permanently above standard application windows
                ns_window.setLevel_(NSFloatingWindowLevel)
                # Display across all desktop Spaces, full-screen apps, and stay stationary
                ns_window.setCollectionBehavior_(
                    CAN_JOIN_ALL_SPACES | FULL_SCREEN_AUX | STATIONARY
                )
        except Exception as exc:
            print(f"[Spokn] Notice configuring macOS window: {exc}", flush=True)

    # -------------------------------------------------------------------------
    # Painting
    # -------------------------------------------------------------------------

    def paintEvent(self, event) -> None:
        """Paint the Dictation Pill capsule, border, shadow, and inner controls."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)

        pill_rect = QRectF(self.PADDING, self.PADDING, self._current_w, self._current_h)
        corner_radius = self._current_h / 2.0

        # 1. Drop shadow & subtle white rim glow in active states; none in idle
        if self._state != PillState.IDLE:
            self._paint_shadow(painter, pill_rect, corner_radius)
            self._paint_rim_glow(painter, pill_rect, corner_radius)

        # 2. Pill capsule fill: solid #0A0A0A
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(self.COLOR_PILL_BG))
        painter.drawRoundedRect(pill_rect, corner_radius, corner_radius)

        # 3. Inside border: crisp 1.2 px solid #FFFFFF (~65% in active, ~45% in IDLE)
        border_alpha = int(255 * 0.45) if self._state == PillState.IDLE else int(255 * 0.65)
        border_pen = QPen(QColor(255, 255, 255, border_alpha), 1.2)
        painter.setPen(border_pen)
        painter.setBrush(Qt.NoBrush)
        # Inset stroke by 0.6px for crisp inside 1.2px stroke
        inset_rect = pill_rect.adjusted(0.6, 0.6, -0.6, -0.6)
        painter.drawRoundedRect(inset_rect, max(0.0, corner_radius - 0.6), max(0.0, corner_radius - 0.6))

        # 4. State-specific interior controls (hidden in IDLE)
        if self._state == PillState.IDLE:
            return

        if self._state in (PillState.LISTENING, PillState.SPEAKING):
            self._paint_listening_speaking_contents(painter, pill_rect)
        elif self._state == PillState.COLLAPSING:
            self._paint_collapsing_contents(painter, pill_rect)
        elif self._state == PillState.PROCESSING:
            self._paint_processing_contents(painter, pill_rect)

    def _paint_rim_glow(self, painter: QPainter, pill_rect: QRectF, radius: float) -> None:
        """Paint 2-3 very tight soft white halo passes (1-3 px spread, low alpha ~10-18%)."""
        painter.save()
        painter.setPen(Qt.NoPen)
        glow_passes = [
            (3.0, int(255 * 0.10)),
            (2.0, int(255 * 0.14)),
            (1.0, int(255 * 0.18)),
        ]
        for spread, alpha in glow_passes:
            painter.setBrush(QColor(255, 255, 255, alpha))
            glow_rect = pill_rect.adjusted(-spread, -spread, spread, spread)
            painter.drawRoundedRect(glow_rect, radius + spread, radius + spread)
        painter.restore()

    def _paint_shadow(self, painter: QPainter, pill_rect: QRectF, radius: float) -> None:
        """Paint soft, low-alpha elevation shadow without visible outer ring artifacts."""
        painter.save()
        painter.setPen(Qt.NoPen)
        blur_steps = 8
        dy = 3.0
        for i in range(blur_steps, 0, -1):
            progress = i / blur_steps
            # Max alpha ~20% with smooth quadratic falloff
            alpha = int(255 * 0.20 * (1.0 - progress) ** 2 / blur_steps * 2.8)
            if alpha <= 0:
                continue
            painter.setBrush(QColor(0, 0, 0, alpha))
            spread = float(i) * 0.85
            shadow_rect = pill_rect.adjusted(-spread, -spread + dy, spread, spread + dy)
            painter.drawRoundedRect(shadow_rect, radius + spread, radius + spread)
        painter.restore()

    def _paint_listening_speaking_contents(self, painter: QPainter, pill_rect: QRectF) -> None:
        """Render Cancel button, 10-bar waveform, and Stop button in 124 × 36 px capsule."""
        cy = pill_rect.center().y()

        # Cancel Button (left): 24 × 24 px circle at x = PADDING + 6
        cancel_center = QPointF(pill_rect.left() + 6.0 + 12.0, cy)
        self._paint_cancel_button(painter, cancel_center, size=24.0, opacity=1.0)

        # Waveform (center): 10 bars, total width 47 px
        wave_left = pill_rect.center().x() - 47.0 / 2.0
        is_active = self._state in (PillState.LISTENING, PillState.SPEAKING)
        self._paint_waveform(painter, wave_left, cy, is_active=is_active)

        # Stop Button (right): 24 × 24 px circle at x = pill_right - 6 - 24
        stop_center = QPointF(pill_rect.right() - 6.0 - 12.0, cy)
        self._paint_stop_button(painter, stop_center, size=24.0, opacity=1.0)

    def _paint_collapsing_contents(self, painter: QPainter, pill_rect: QRectF) -> None:
        """Render 4 px shrunk edge dots and flat waveform in 100 × 28 px capsule."""
        cy = pill_rect.center().y()

        # Shrunk Cancel dot (left): 4 px dot at 25% opacity
        cancel_center = QPointF(pill_rect.left() + 10.0 + 2.0, cy)
        self._paint_cancel_button(painter, cancel_center, size=4.0, opacity=0.25)

        # Waveform: flat 2 px bars
        wave_left = pill_rect.center().x() - 47.0 / 2.0
        self._paint_waveform(painter, wave_left, cy, is_active=False)

        # Shrunk Stop dot (right): 4 px coral red dot at 25% opacity
        stop_center = QPointF(pill_rect.right() - 10.0 - 2.0, cy)
        self._paint_stop_button(painter, stop_center, size=4.0, opacity=0.25)

    def _paint_processing_contents(self, painter: QPainter, pill_rect: QRectF) -> None:
        """Render flat waveform and 3 pulsing dots centered in 90 × 26 px capsule.

        Combined group width: 47 px (waveform) + 7 px (gap) + 18 px (3 dots) = 72 px.
        Centering horizontally leaves exactly (90 - 72) / 2 = 9 px margin on both sides.
        """
        cy = pill_rect.center().y()
        start_x = pill_rect.center().x() - 36.0

        # Left: flat 10-bar waveform (47 px width)
        self._paint_waveform(painter, start_x, cy, is_active=False)

        # Right: 3 sequential pulsing dots (total 18 px span: 3 dots of 4 px with 3 px gaps)
        dots_start_x = start_x + 47.0 + 7.0
        for i in range(3):
            dot_center = QPointF(dots_start_x + 2.0 + i * 7.0, cy)
            is_active = (i == self._processing_phase)
            color = self.COLOR_WHITE if is_active else self.COLOR_DOT_INACTIVE
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(color))
            painter.drawEllipse(dot_center, 2.0, 2.0)

    def _paint_cancel_button(self, painter: QPainter, center: QPointF, size: float, opacity: float) -> None:
        """Paint Cancel button or shrunk dot."""
        painter.save()
        radius = size / 2.0

        # Circle background
        face_color = QColor(self.COLOR_CANCEL_BG)
        face_color.setAlpha(int(face_color.alpha() * opacity))
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(face_color))
        painter.drawEllipse(center, radius, radius)

        # If full size (24 px), draw white '×' icon
        if size >= 16.0:
            pen = QPen(self.COLOR_WHITE, 1.5, Qt.SolidLine, Qt.RoundCap)
            painter.setPen(pen)
            r = 3.5
            painter.drawLine(QPointF(center.x() - r, center.y() - r), QPointF(center.x() + r, center.y() + r))
            painter.drawLine(QPointF(center.x() - r, center.y() + r), QPointF(center.x() + r, center.y() - r))

        painter.restore()

    def _paint_stop_button(self, painter: QPainter, center: QPointF, size: float, opacity: float) -> None:
        """Paint Stop button with coral red circle and centered white square."""
        painter.save()
        radius = size / 2.0

        # Red circle background
        face_color = QColor(self.COLOR_STOP_RED)
        face_color.setAlpha(int(255 * opacity))
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(face_color))
        painter.drawEllipse(center, radius, radius)

        # If full size (24 px), draw 7.5 × 7.5 px white square (1.5 px corner radius)
        if size >= 16.0:
            square_size = 7.5
            sq_rect = QRectF(
                center.x() - square_size / 2.0,
                center.y() - square_size / 2.0,
                square_size,
                square_size,
            )
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(self.COLOR_WHITE))
            painter.drawRoundedRect(sq_rect, 1.5, 1.5)

        painter.restore()

    def _paint_waveform(self, painter: QPainter, start_x: float, cy: float, is_active: bool) -> None:
        """Paint 10 vertical white bars (2 px width, 3 px gap, 1 px radius)."""
        painter.save()
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(self.COLOR_WHITE))

        bar_width = 2.0
        bar_gap = 3.0

        heights = self._bar_heights if is_active else self.WAVEFORM_FLAT

        for i, h in enumerate(heights):
            x = start_x + i * (bar_width + bar_gap)
            y = cy - h / 2.0
            bar_rect = QRectF(x, y, bar_width, h)
            painter.drawRoundedRect(bar_rect, 1.0, 1.0)

        painter.restore()

    # -------------------------------------------------------------------------
    # Mouse Interaction & Hit-Testing
    # -------------------------------------------------------------------------

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Handle mouse clicks on the pill and edge buttons."""
        if event.button() != Qt.LeftButton:
            return

        pos = event.position()
        pill_rect = QRectF(self.PADDING, self.PADDING, self._current_w, self._current_h)
        if not pill_rect.contains(pos):
            return

        # Idle click
        if self._state == PillState.IDLE:
            self.idle_clicked.emit()
            return

        # Edge button hit-testing in Listening / Speaking
        if self._state in (PillState.LISTENING, PillState.SPEAKING):
            cy = pill_rect.center().y()
            cancel_center = QPointF(pill_rect.left() + 18.0, cy)
            stop_center = QPointF(pill_rect.right() - 18.0, cy)

            # Hit radius of 14 px around 24 px buttons
            if (pos - cancel_center).manhattanLength() <= 16.0:
                self.cancel_clicked.emit()
            elif (pos - stop_center).manhattanLength() <= 16.0:
                self.stop_clicked.emit()


# -----------------------------------------------------------------------------
# Standalone Preview Mode
# -----------------------------------------------------------------------------

def main() -> None:
    """Launch standalone preview cycling through all 5 Figma visual states."""
    app = QApplication.instance() or QApplication(sys.argv)

    overlay = DictationPillOverlay()
    overlay.show()

    # Clean exit on Ctrl+C in terminal
    signal.signal(signal.SIGINT, lambda *_: app.quit())

    # Sequence of states to cycle through
    sequence = [
        (PillState.IDLE, 2000, "1/5: IDLE (36 × 6 px)"),
        (PillState.LISTENING, 2000, "2/5: LISTENING (124 × 36 px)"),
        (PillState.SPEAKING, 2500, "3/5: SPEAKING (124 × 36 px, active waveform)"),
        (PillState.COLLAPSING, 220, "4/5: COLLAPSING (100 × 28 px) -> auto PROCESSING"),
        (PillState.PROCESSING, 2200, "5/5: PROCESSING (90 × 26 px, pulsing dots)"),
    ]

    seq_idx = 0

    def step() -> None:
        nonlocal seq_idx
        state, duration, desc = sequence[seq_idx]
        print(f"[Spokn Preview] State: {desc}")
        overlay.set_state(state)
        seq_idx = (seq_idx + 1) % len(sequence)
        QTimer.singleShot(duration, step)

    # Wire interactive clicks
    overlay.idle_clicked.connect(lambda: print("[Spokn Preview] Clicked IDLE pill"))
    overlay.cancel_clicked.connect(lambda: print("[Spokn Preview] Clicked CANCEL button"))
    overlay.stop_clicked.connect(lambda: print("[Spokn Preview] Clicked STOP button"))

    # Start first step
    QTimer.singleShot(500, step)

    # Heartbeat timer so Python SIGINT can be handled by Qt event loop
    sigint_timer = QTimer()
    sigint_timer.timeout.connect(lambda: None)
    sigint_timer.start(200)

    print("[Spokn Preview] Preview running. Press Ctrl+C in terminal to exit.")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
