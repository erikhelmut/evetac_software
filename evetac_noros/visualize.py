"""Live Evetac visualizer (OpenCV, no ROS).

Tracking mode (default):
  left:  events with the tracked dots and their (magnified) displacement from the reference
  right: event rate, dot displacement magnitude, mean shift (shear)
Raw mode (--raw): no tracker, no noise filter (unless --noise-filter-ms is given)
  left:  events only, as ON/OFF polarity or per-pixel event count heatmap
  right: event rate, active pixels, event count in 1 ms bins (shows vibrations up to 500 Hz)

Usage:
    python -m evetac_noros.visualize                # first connected camera
    python -m evetac_noros.visualize --raw          # raw events only
    python -m evetac_noros.visualize --synthetic    # no hardware needed
    python -m evetac_noros.visualize --file rec.aedat4

Keys: q/Esc quit | space pause display | m polarity/count view | d accumulation window | c clear plots
      h help | tracking mode only: t toggle tracking | r current dots -> reference | s store calibration
      +/- arrow gain
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
FAST_HISTORY_S = 1.0
LUT_LEVELS = 32
WINDOWS_MS = (0, 50, 200, 1000)  # event accumulation time constants; 0 = latest frame only


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


def _ramp(c0, c1, n=LUT_LEVELS):
    a = (np.arange(n) / (n - 1))[:, None]
    return (np.array(c0) * (1 - a) + np.array(c1) * a).astype(np.uint8)


class Plot:
    def __init__(self, title, unit, series, symmetric=False, history_s=HISTORY_S, tick_s=2.0, thickness=2,
                 fmt="{:.1f}"):
        self.title, self.unit, self.series, self.symmetric = title, unit, series, symmetric
        self.history_s, self.tick_s, self.thickness, self.fmt = history_s, tick_s, thickness, fmt

    def draw(self, img, x0, y0, w, h, t, values, now):
        """series: list of (key, label, color); values: dict key -> array aligned with t."""
        _text(img, self.title, (x0, y0 + 14), INK, 0.48)
        tw = cv2.getTextSize(self.title, FONT, 0.48, 1)[0][0]
        _text(img, self.unit, (x0 + tw + 8, y0 + 14), MUTED, 0.4)
        # legend with current values: chip + label for >= 2 series, only the value for a single series
        lx = x0 + w
        for key, label, color in reversed(self.series):
            v = values[key][-1] if len(values[key]) else 0.0
            s = self.fmt.format(v) if len(self.series) == 1 else f"{label} {self.fmt.format(v)}"
            tw = cv2.getTextSize(s, FONT, 0.4, 1)[0][0]
            if len(self.series) == 1:
                lx -= tw
                _text(img, s, (lx, y0 + 14), INK_2, 0.4)
                continue
            lx -= tw + 22
            cv2.rectangle(img, (lx, y0 + 5), (lx + 10, y0 + 15), color, -1)
            _text(img, s, (lx + 14, y0 + 14), INK_2, 0.4)

        top, bottom = y0 + 26, y0 + h - 18
        left, right = x0 + 44, x0 + w
        mask = t >= now - self.history_s
        vis = [values[key][mask] for key, _, _ in self.series]
        vmax = max([np.abs(v).max() for v in vis if len(v)] + [0])
        ymax = _nice_max(vmax * 1.1 if vmax > 0 else 1.0)
        ymin = -ymax if self.symmetric else 0.0

        def ypix(v):
            return (bottom - (np.asarray(v) - ymin) / (ymax - ymin) * (bottom - top)).astype(np.int32)

        for frac in (0, 0.5, 1):
            yv = ymin + frac * (ymax - ymin)
            yp = int(ypix(yv))
            cv2.line(img, (left, yp), (right, yp), AXIS if yv == 0 else GRID, 1)
            _text(img, f"{yv:g}", (x0, yp + 4), MUTED, 0.36)
        for s in np.arange(0, self.history_s + 1e-9, self.tick_s):
            xp = int(right - s / self.history_s * (right - left))
            cv2.line(img, (xp, bottom), (xp, bottom + 3), AXIS, 1)
            label = "now" if s == 0 else f"-{s:g}s"
            _text(img, label, (xp - (22 if s == 0 else 12), bottom + 15), MUTED, 0.34)

        tt = t[mask]
        if len(tt) < 2:
            return
        xs = (right - (now - tt) / self.history_s * (right - left)).astype(np.int32)
        for (_, _, color), v in zip(self.series, vis):
            pts = np.stack([xs, np.clip(ypix(v), top, bottom)], axis=1)
            cv2.polylines(img, [pts], False, color, self.thickness, cv2.LINE_AA)


class Visualizer:
    def __init__(self, reader, arrow_gain=4.0, window_ms=50, count_view=None):
        self.reader = reader
        self.raw = reader.tracker is None
        self.res = reader.resolution
        self.scale = PANEL_H / self.res[1]
        self.panel_w = int(round(self.res[0] * self.scale))
        self.arrow_gain = arrow_gain
        self.window_ms = window_ms
        self.count_view = self.raw if count_view is None else count_view
        self.show_help = False
        self.paused = False
        self.message, self.message_until = "", 0.0
        self.frame_ms = 1000.0 / reader.rate_hz

        n_pix = self.res[0] * self.res[1]
        self.on_acc = np.zeros(n_pix, np.float32)
        self.off_acc = np.zeros(n_pix, np.float32)
        self.cmax = 2.0  # auto-scaled upper end of the color scale [events per pixel]
        # color lookups: polarity view index = level + LUT_LEVELS * (OFF dominates); count view = level
        self.lut_polarity = np.concatenate([_ramp(SURFACE, C_ON), _ramp(SURFACE, C_OFF)])
        self.lut_count = _ramp(SURFACE, INK)

        if self.raw:
            self.plots = [
                Plot("Event rate", "(k events/s)", [("on", "ON", C_ON), ("off", "OFF", C_OFF)]),
                Plot("Active pixels", "(% of sensor per frame)", [("active", "", C_MEAN)], fmt="{:.2f}"),
                Plot("Events per 1 ms", "(count, last 1 s)", [("fast", "", C_MAX)],
                     history_s=FAST_HISTORY_S, tick_s=0.2, thickness=1, fmt="{:.0f}"),
            ]
        else:
            self.plots = [
                Plot("Event rate", "(k events/s)", [("on", "ON", C_ON), ("off", "OFF", C_OFF)]),
                Plot("Dot displacement", "(px)", [("mean", "mean", C_MEAN), ("max", "max", C_MAX)]),
                Plot("Mean shift", "(px, shear)", [("x", "x", C_DX), ("y", "y", C_DY)], symmetric=True),
            ]
        self.clear_history()
        self.last_frame = None
        self.fps = 0.0

    def clear_history(self):
        n = int(HISTORY_S * self.reader.rate_hz * 1.2)
        self.hist = {k: deque(maxlen=n) for k in ("t", "on", "off", "active", "mean", "max", "x", "y")}
        n_fast = int(FAST_HISTORY_S * 1000 * 1.2)
        self.fast = {k: deque(maxlen=n_fast) for k in ("t", "fast")}

    def notify(self, msg):
        self.message, self.message_until = msg, time.time() + 2.5
        print(msg)

    # --------------------------------------------------------------- data
    def update(self, frame):
        ev = frame.events
        pos = ev["polarity"] > 0
        flat = ev["y"].astype(np.int64) * self.res[0] + ev["x"]
        n_pix = self.on_acc.size
        on_now = np.bincount(flat[pos], minlength=n_pix)
        off_now = np.bincount(flat[~pos], minlength=n_pix)
        decay = np.exp(-self.frame_ms / self.window_ms) if self.window_ms > 0 else 0.0
        self.on_acc *= decay
        self.off_acc *= decay
        self.on_acc += on_now
        self.off_acc += off_now

        n_on = int(np.count_nonzero(pos))
        h = self.hist
        h["t"].append(frame.timestamp * 1e-6)
        h["on"].append(n_on / self.frame_ms)  # events per ms = k events / s
        h["off"].append((len(ev) - n_on) / self.frame_ms)
        if self.raw:
            h["active"].append(100.0 * np.count_nonzero(on_now + off_now) / n_pix)
            n_bins = max(1, int(round((frame.timestamp - frame.start_timestamp) / 1000)))
            bins = np.clip((ev["timestamp"] - frame.start_timestamp) // 1000, 0, n_bins - 1)
            counts = np.bincount(bins, minlength=n_bins)
            self.fast["t"].extend((frame.start_timestamp + 1000 * (np.arange(n_bins) + 1)) * 1e-6)
            self.fast["fast"].extend(counts)
        else:
            disp = frame.displacement_xy
            mag = np.linalg.norm(disp, axis=1)
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

    def _levels(self, total):
        """Map event counts to LUT levels with a log scale that follows the recent maximum."""
        active = total[total > 0.05]
        target = np.percentile(active, 99) if active.size > 50 else 2.0
        self.cmax = max(1.0, 0.8 * self.cmax + 0.2 * target)
        level = np.log1p(total) / np.log1p(self.cmax) * (LUT_LEVELS - 1)
        return np.minimum(level, LUT_LEVELS - 1).astype(np.uint8)

    def _event_panel(self):
        total = self.on_acc + self.off_acc
        level = self._levels(total)
        if self.count_view:
            img = self.lut_count[level]
        else:
            img = self.lut_polarity[level + LUT_LEVELS * (self.off_acc > self.on_acc)]
        img = img.reshape(self.res[1], self.res[0], 3)
        if self.scale != 1:
            img = cv2.resize(img, (self.panel_w, PANEL_H), interpolation=cv2.INTER_NEAREST)

        f = self.last_frame
        window = "latest frame" if self.window_ms == 0 else f"~{self.window_ms} ms window"
        if self.raw:
            title = f"Raw events, {'count' if self.count_view else 'polarity'} ({window})"
        else:
            title = f"Events + tracked dots, arrows x{self.arrow_gain:g} ({window})"
            if f is not None:
                self._draw_dots(img, f)
        _text(img, title, (10, 20), INK, 0.48)
        if f is not None and not self.raw and not f.tracking_active:
            _text(img, "TRACKING PAUSED", (10, 42), C_ALERT, 0.5, 1)
        self._panel_legend(img)
        return img

    def _draw_dots(self, img, f):
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

    def _panel_legend(self, img):
        y = PANEL_H - 12
        x = 10
        if self.count_view:
            bar_w = 120
            bar = np.repeat(self.lut_count[np.linspace(0, LUT_LEVELS - 1, bar_w).astype(int)][None], 10, axis=0)
            img[y - 10:y, x + 12:x + 12 + bar_w] = bar
            cv2.rectangle(img, (x + 11, y - 11), (x + 13 + bar_w, y + 1), AXIS, 1)
            _text(img, "0", (x, y), INK_2, 0.4)
            _text(img, f"{self.cmax:.0f} events/px (log)", (x + bar_w + 18, y), INK_2, 0.4)
            x += bar_w + 190
        else:
            for label, color in (("ON", C_ON), ("OFF", C_OFF)):
                cv2.rectangle(img, (x, y - 10), (x + 10, y), color, -1)
                _text(img, label, (x + 14, y), INK_2, 0.4)
                x += 50
            x += 20
        if not self.raw:
            cv2.circle(img, (x, y - 5), 3, MUTED, -1, cv2.LINE_AA)
            _text(img, "reference", (x + 8, y), INK_2, 0.4)
            cv2.circle(img, (x + 98, y - 5), 4, INK_2, 1, cv2.LINE_AA)
            _text(img, "tracked", (x + 110, y), INK_2, 0.4)

    def _plots(self, side):
        if not self.hist["t"]:
            _text(side, "waiting for events...", (16, 30), MUTED, 0.5)
            return
        t = np.fromiter(self.hist["t"], float)
        values = {k: np.fromiter(v, float) for k, v in self.hist.items() if k != "t"}
        t_fast = np.fromiter(self.fast["t"], float)
        values["fast"] = np.fromiter(self.fast["fast"], float)
        ph = (PANEL_H - 16) // 3
        for i, plot in enumerate(self.plots):
            tt = t_fast if plot.series[0][0] == "fast" else t
            if len(tt):
                plot.draw(side, 14, 8 + i * ph, PLOT_W - 28, ph - 6, tt, values, t[-1])

    def _status(self, canvas, H):
        r = self.reader
        s = f"{r.source.name}  {r.resolution[0]}x{r.resolution[1]}  {r.rate_hz:.0f} Hz frames  |  "
        if self.raw:
            nf = getattr(r.source, "noise_filter", None) is not None
            s += f"raw: no tracking, noise filter {'on' if nf else 'off'}"
        else:
            s += f"{r.slice_ms} ms tracker slices, slow {r.slow_slices}/{r.total_slices}"
        s += f"  |  display {self.fps:4.1f} fps"
        if self.paused:
            s = "[PAUSED]  " + s
        _text(canvas, s, (16, H - 14), MUTED, 0.4)
        help_s = self.message if time.time() < self.message_until else "h: help"
        tw = cv2.getTextSize(help_s, FONT, 0.4, 1)[0][0]
        _text(canvas, help_s, (canvas.shape[1] - 16 - tw, H - 14), INK_2, 0.4)
        if self.show_help:
            lines = ["q / Esc   quit", "space     pause display", "m         polarity / count view",
                     "d         accumulation window", "c         clear plots"]
            if not self.raw:
                lines += ["t         toggle tracking", "r         current dots -> reference",
                          "s         store calibration (.pkl)", "+ / -     arrow gain"]
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
        elif key == ord("m"):
            self.count_view = not self.count_view
        elif key == ord("d"):
            self.window_ms = WINDOWS_MS[(WINDOWS_MS.index(self.window_ms) + 1) % len(WINDOWS_MS)] \
                if self.window_ms in WINDOWS_MS else WINDOWS_MS[0]
            self.notify("accumulation: " + ("latest frame" if self.window_ms == 0 else f"~{self.window_ms} ms"))
        elif key == ord("c"):
            self.clear_history()
        elif key == ord("h"):
            self.show_help = not self.show_help
        elif key in (ord("t"), ord("r"), ord("s"), ord("+"), ord("="), ord("-"), ord("_")) and self.raw:
            self.notify("no dot tracking in raw mode")
        elif key == ord("t"):
            r.tracking_active = not r.tracking_active
            self.notify(f"tracking {'on' if r.tracking_active else 'off'}")
        elif key == ord("r"):
            r.reset_reference()
            self.notify("current dot positions are the new reference")
        elif key == ord("s"):
            path = r.tracker.store_current_calibration()
            self.notify(f"stored {path.rsplit('/', 1)[-1]}")
        elif key in (ord("+"), ord("=")):
            self.arrow_gain = min(self.arrow_gain * 1.5, 50)
        elif key in (ord("-"), ord("_")):
            self.arrow_gain = max(self.arrow_gain / 1.5, 1)
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
    parser.add_argument("--window-ms", type=int, default=50, choices=WINDOWS_MS,
                        help="event accumulation time constant for the image, 0 = latest frame only")
    parser.add_argument("--view", choices=("polarity", "count"), default=None,
                        help="event image style (default: count in raw mode, polarity otherwise)")
    args = parser.parse_args()
    reader = EvetacReader(make_source(args), args.calibration, rate_hz=args.rate, crop=args.crop, track=not args.raw)
    count_view = None if args.view is None else args.view == "count"
    Visualizer(reader, arrow_gain=args.arrow_gain, window_ms=args.window_ms, count_view=count_view).run()


if __name__ == "__main__":
    main()
