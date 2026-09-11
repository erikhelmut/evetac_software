"""Live Evetac visualizer (OpenCV, no ROS).

Left:  events of the latest frames (ON / OFF) with the tracked dots and their displacement
       from the reference (calibration) position, arrows magnified.
Right: rolling plots of the event rate, dot displacement magnitude and mean shift (shear).

Usage:
    python -m evetac_noros.visualize                # first connected camera
    python -m evetac_noros.visualize --synthetic    # no hardware needed
    python -m evetac_noros.visualize --file rec.aedat4

Keys: q/Esc quit | space pause display | t toggle tracking | r use current dots as reference
      s store calibration | c clear plots | +/- arrow gain | d toggle persistence | h help
"""

import argparse
import time
from collections import deque

import cv2
import numpy as np

from .reader import EvetacReader, add_common_args, make_source


def _bgr(hex_color):
    h = hex_color.lstrip("#")
    return int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16)


# dark chart palette (validated categorical slots, see dataviz reference palette)
SURFACE = _bgr("#1a1a19")
PLANE = _bgr("#0d0d0d")
INK = _bgr("#ffffff")
INK_2 = _bgr("#c3c2b7")
MUTED = _bgr("#898781")
GRID = _bgr("#2c2c2a")
AXIS = _bgr("#383835")
C_ON = _bgr("#3987e5")
C_OFF = _bgr("#d95926")
C_MEAN = _bgr("#199e70")
C_MAX = _bgr("#c98500")
C_DX = _bgr("#d55181")
C_DY = _bgr("#9085e9")
C_ALERT = _bgr("#d03b3b")

FONT = cv2.FONT_HERSHEY_SIMPLEX
PANEL_H = 480
PLOT_W = 460
HISTORY_S = 10.0


def _text(img, s, org, color=INK_2, scale=0.42, thickness=1):
    cv2.putText(img, s, org, FONT, scale, color, thickness, cv2.LINE_AA)


def _nice_max(v):
    if v <= 0:
        return 1.0
    exp = 10 ** np.floor(np.log10(v))
    for m in (1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10):
        if v <= m * exp:
            return m * exp
    return 10 * exp


class Plot:
    def __init__(self, title, unit, series, symmetric=False):
        self.title, self.unit, self.series, self.symmetric = title, unit, series, symmetric

    def draw(self, img, x0, y0, w, h, t, values, now):
        """series: list of (label, color); values: dict label -> array aligned with t."""
        _text(img, self.title, (x0, y0 + 14), INK, 0.48)
        tw = cv2.getTextSize(self.title, FONT, 0.48, 1)[0][0]
        _text(img, self.unit, (x0 + tw + 8, y0 + 14), MUTED, 0.4)
        # legend with current values (identity by chip + label, value in ink)
        lx = x0 + w
        for label, color in reversed(self.series):
            v = values[label][-1] if len(values[label]) else 0.0
            s = f"{label} {v:.1f}"
            tw = cv2.getTextSize(s, FONT, 0.4, 1)[0][0]
            lx -= tw + 22
            cv2.rectangle(img, (lx, y0 + 5), (lx + 10, y0 + 15), color, -1)
            _text(img, s, (lx + 14, y0 + 14), INK_2, 0.4)

        top, bottom = y0 + 26, y0 + h - 18
        left, right = x0 + 44, x0 + w
        mask = t >= now - HISTORY_S
        vis = [values[label][mask] for label, _ in self.series]
        vmax = max([np.abs(v).max() for v in vis if len(v)] + [0])
        ymax = _nice_max(vmax * 1.1 if vmax > 0 else 1.0)
        ymin = -ymax if self.symmetric else 0.0

        def ypix(v):
            return (bottom - (np.asarray(v) - ymin) / (ymax - ymin) * (bottom - top)).astype(np.int32)

        for frac in (0, 0.5, 1):
            yv = ymin + frac * (ymax - ymin)
            yp = int(ypix(yv))
            is_base = (yv == 0)
            cv2.line(img, (left, yp), (right, yp), AXIS if is_base else GRID, 1)
            _text(img, f"{yv:g}", (x0, yp + 4), MUTED, 0.36)
        for s in range(0, int(HISTORY_S) + 1, 2):
            xp = int(right - s / HISTORY_S * (right - left))
            cv2.line(img, (xp, bottom), (xp, bottom + 3), AXIS, 1)
            _text(img, "now" if s == 0 else f"-{s}s", (xp - (22 if s == 0 else 10), bottom + 15), MUTED, 0.34)

        tt = t[mask]
        if len(tt) < 2:
            return
        xs = (right - (now - tt) / HISTORY_S * (right - left)).astype(np.int32)
        for (label, color), v in zip(self.series, vis):
            pts = np.stack([xs, np.clip(ypix(v), top, bottom)], axis=1)
            cv2.polylines(img, [pts], False, color, 2, cv2.LINE_AA)


class Visualizer:
    def __init__(self, reader, arrow_gain=4.0, persistence=True):
        self.reader = reader
        self.res = reader.resolution
        self.scale = PANEL_H / self.res[1]
        self.panel_w = int(round(self.res[0] * self.scale))
        self.arrow_gain = arrow_gain
        self.persistence = persistence
        self.show_help = False
        self.paused = False
        self.message, self.message_until = "", 0.0
        self.on_acc = np.zeros(self.res[1] * self.res[0], np.float32)
        # color lookup: index = level (0..15) + 16 * (OFF dominates)
        a = (np.arange(16) / 15.0)[:, None]
        self.lut = np.concatenate([np.array(SURFACE) * (1 - a) + np.array(C_ON) * a,
                                   np.array(SURFACE) * (1 - a) + np.array(C_OFF) * a]).astype(np.uint8)
        self.off_acc = np.zeros_like(self.on_acc)
        self.clear_history()
        self.plots = [
            Plot("Event rate", "(k events/s)", [("ON", C_ON), ("OFF", C_OFF)]),
            Plot("Dot displacement", "(px)", [("mean", C_MEAN), ("max", C_MAX)]),
            Plot("Mean shift", "(px, shear)", [("x", C_DX), ("y", C_DY)], symmetric=True),
        ]
        self.last_frame = None
        self.fps = 0.0

    def clear_history(self):
        n = int(HISTORY_S * self.reader.rate_hz * 1.2)
        self.hist = {k: deque(maxlen=n) for k in ("t", "ON", "OFF", "mean", "max", "x", "y")}

    def notify(self, msg):
        self.message, self.message_until = msg, time.time() + 2.5
        print(msg)

    # --------------------------------------------------------------- data
    def update(self, frame):
        ev = frame.events
        pos = ev["polarity"] > 0
        flat = ev["y"].astype(np.int64) * self.res[0] + ev["x"]
        n_pix = self.on_acc.size
        if self.persistence:
            self.on_acc *= 0.6
            self.off_acc *= 0.6
        else:
            self.on_acc[:] = 0
            self.off_acc[:] = 0
        self.on_acc += np.bincount(flat[pos], minlength=n_pix)
        self.off_acc += np.bincount(flat[~pos], minlength=n_pix)

        dt = 1.0 / self.reader.rate_hz
        disp = frame.displacement_xy
        mag = np.linalg.norm(disp, axis=1)
        h = self.hist
        h["t"].append(frame.timestamp * 1e-6)
        h["ON"].append(np.count_nonzero(pos) / dt / 1e3)
        h["OFF"].append((len(ev) - np.count_nonzero(pos)) / dt / 1e3)
        h["mean"].append(mag.mean())
        h["max"].append(mag.max())
        h["x"].append(disp[:, 0].mean())
        h["y"].append(disp[:, 1].mean())
        self.last_frame = frame

    # ------------------------------------------------------------- render
    def render(self):
        W = self.panel_w + PLOT_W + 3 * 16
        H = PANEL_H + 2 * 16 + 30
        canvas = np.full((H, W, 3), PLANE, np.uint8)
        canvas[16:16 + PANEL_H, 16:16 + self.panel_w] = self._event_panel()
        side = canvas[16:16 + PANEL_H, 32 + self.panel_w:32 + self.panel_w + PLOT_W]
        side[:] = SURFACE
        self._plots(side)
        self._status(canvas, H)
        return canvas

    def _event_panel(self):
        level = np.minimum(np.maximum(self.on_acc, self.off_acc) * 7.5, 15).astype(np.uint8)
        idx = level + 16 * (self.off_acc > self.on_acc).astype(np.uint8)
        img = self.lut[idx].reshape(self.res[1], self.res[0], 3)
        if self.scale != 1:
            img = cv2.resize(img, (self.panel_w, PANEL_H), interpolation=cv2.INTER_NEAREST)

        f = self.last_frame
        if f is not None:
            sub = 4  # draw with sub-pixel precision
            ref = f.initial_dots_xy * self.scale
            cur = f.dots_xy * self.scale
            tip = ref + (cur - ref) * self.arrow_gain
            for p0, p1, pc in zip(ref, tip, cur):
                cv2.circle(img, tuple(np.round(p0 * 2 ** sub).astype(int)), 2 << sub, MUTED, -1, cv2.LINE_AA, sub)
                cv2.circle(img, tuple(np.round(pc * 2 ** sub).astype(int)), 4 << sub, INK_2, 1, cv2.LINE_AA, sub)
                if np.hypot(*(p1 - p0)) > 1.5:
                    cv2.arrowedLine(img, tuple(np.round(p0).astype(int)), tuple(np.round(p1).astype(int)),
                                    INK, 2, cv2.LINE_AA, tipLength=0.25)
        _text(img, f"Events + tracked dots (arrows x{self.arrow_gain:g})", (10, 20), INK, 0.48)
        if f is not None and not f.tracking_active:
            _text(img, "TRACKING PAUSED", (10, 42), C_ALERT, 0.5, 1)
        # polarity legend
        y = PANEL_H - 12
        for label, color, x in (("ON", C_ON, 10), ("OFF", C_OFF, 60)):
            cv2.rectangle(img, (x, y - 10), (x + 10, y), color, -1)
            _text(img, label, (x + 14, y), INK_2, 0.4)
        cv2.circle(img, (130, y - 5), 3, MUTED, -1, cv2.LINE_AA)
        _text(img, "reference", (138, y), INK_2, 0.4)
        cv2.circle(img, (228, y - 5), 4, INK_2, 1, cv2.LINE_AA)
        _text(img, "tracked", (240, y), INK_2, 0.4)
        return img

    def _plots(self, side):
        if not self.hist["t"]:
            _text(side, "waiting for events...", (16, 30), MUTED, 0.5)
            return
        t = np.fromiter(self.hist["t"], float)
        values = {k: np.fromiter(v, float) for k, v in self.hist.items() if k != "t"}
        ph = (PANEL_H - 16) // 3
        for i, plot in enumerate(self.plots):
            plot.draw(side, 14, 8 + i * ph, PLOT_W - 28, ph - 6, t, values, t[-1])

    def _status(self, canvas, H):
        r = self.reader
        s = (f"{r.source.name}  {r.resolution[0]}x{r.resolution[1]}  {r.rate_hz:.0f} Hz frames, "
             f"{r.slice_ms} ms tracker slices  |  display {self.fps:4.1f} fps  |  "
             f"slow slices {r.slow_slices}/{r.total_slices}")
        if self.paused:
            s = "[PAUSED]  " + s
        _text(canvas, s, (16, H - 14), MUTED, 0.4)
        help_s = "h: help"
        if time.time() < self.message_until:
            help_s = self.message
        tw = cv2.getTextSize(help_s, FONT, 0.4, 1)[0][0]
        _text(canvas, help_s, (canvas.shape[1] - 16 - tw, H - 14), INK_2, 0.4)
        if self.show_help:
            lines = ["q / Esc   quit", "space     pause display", "t         toggle tracking",
                     "r         current dots -> reference", "s         store calibration (.pkl)",
                     "c         clear plots", "+ / -     arrow gain", "d         toggle event persistence"]
            x0, y0 = 32, 60
            cv2.rectangle(canvas, (x0 - 10, y0 - 22), (x0 + 330, y0 + 22 * len(lines) - 8), PLANE, -1)
            for i, line in enumerate(lines):
                _text(canvas, line, (x0, y0 + i * 22), INK, 0.45)

    # --------------------------------------------------------------- keys
    def handle_key(self, key):
        r = self.reader
        if key in (ord("q"), 27):
            return False
        if key == ord(" "):
            self.paused = not self.paused
        elif key == ord("t"):
            r.tracking_active = not r.tracking_active
            self.notify(f"tracking {'on' if r.tracking_active else 'off'}")
        elif key == ord("r"):
            r.reset_reference()
            self.notify("current dot positions are the new reference")
        elif key == ord("s"):
            path = r.tracker.store_current_calibration()
            self.notify(f"stored {path.rsplit('/', 1)[-1]}")
        elif key == ord("c"):
            self.clear_history()
        elif key in (ord("+"), ord("=")):
            self.arrow_gain = min(self.arrow_gain * 1.5, 50)
        elif key in (ord("-"), ord("_")):
            self.arrow_gain = max(self.arrow_gain / 1.5, 1)
        elif key == ord("d"):
            self.persistence = not self.persistence
        elif key == ord("h"):
            self.show_help = not self.show_help
        return True

    def run(self, window="Evetac"):
        cv2.namedWindow(window, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        last_draw, n_draws, fps_t0 = 0.0, 0, time.perf_counter()
        try:
            while self.reader.source.is_running():
                frame = self.reader.poll()
                if frame is not None and not self.paused:
                    self.update(frame)
                now = time.perf_counter()
                if now - last_draw < 1 / 30:  # cap display at ~30 fps, keep reading events
                    continue
                last_draw = now
                cv2.imshow(window, self.render())
                n_draws += 1
                if now - fps_t0 > 1:
                    self.fps, n_draws, fps_t0 = n_draws / (now - fps_t0), 0, now
                if not self.handle_key(cv2.waitKey(1) & 0xFF):
                    break
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break
        except KeyboardInterrupt:
            pass
        finally:
            cv2.destroyAllWindows()
            self.reader.close()


def main():
    parser = argparse.ArgumentParser(description="Live Evetac visualizer (no ROS).")
    add_common_args(parser)
    parser.add_argument("--arrow-gain", type=float, default=4.0, help="magnification of the displacement arrows")
    args = parser.parse_args()
    reader = EvetacReader(make_source(args), args.calibration, rate_hz=args.rate, crop=args.crop)
    Visualizer(reader, arrow_gain=args.arrow_gain).run()


if __name__ == "__main__":
    main()
