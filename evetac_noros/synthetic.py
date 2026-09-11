"""Synthetic Evetac event source, so the reader and visualizer can be tried without hardware.

Dark dots at the calibration locations are displaced by a scripted contact
(idle -> press -> shear -> release). A simple DVS model emits an ON/OFF event whenever a
pixel's log intensity changes by more than a contrast threshold. This is only a rough
approximation of the real sensor, meant for testing the pipeline.
"""

import pickle
import time

import numpy as np

from .reader import EVENT_DTYPE

CYCLE_S = 6.0


def _smoothstep(a, b, t):
    s = np.clip((t - a) / (b - a), 0.0, 1.0)
    return s * s * (3 - 2 * s)


class SyntheticSource:
    def __init__(self, calibration, resolution=(640, 480), dot_radius=9.0, step_ms=2, contrast=0.2,
                 noise_rate_hz=3000, realtime=True, seed=0):
        centers_xy, _ = pickle.load(open(calibration, "rb"))
        self.base = np.asarray(centers_xy, dtype=np.float64).reshape(-1, 2)
        self.name = "synthetic"
        self.resolution = tuple(resolution)
        self.dot_radius = dot_radius
        self.step_us = int(step_ms * 1000)
        self.contrast = contrast
        self.noise_rate_hz = noise_rate_hz
        self.realtime = realtime
        self.rng = np.random.default_rng(seed)

        half = 20  # patch around each dot; dot spacing is ~62 px so patches never overlap
        off = np.arange(-half, half + 1)
        oy, ox = np.meshgrid(off, off, indexing="ij")
        origin = np.round(self.base).astype(np.int64)
        self.px = (origin[:, 0, None, None] + ox[None]).astype(np.int16)  # (N, P, P) pixel x
        self.py = (origin[:, 1, None, None] + oy[None]).astype(np.int16)
        self._pxf = self.px.astype(np.float32)
        self._pyf = self.py.astype(np.float32)
        self.ref = self._log_intensity(np.zeros_like(self.base))
        self.t_us = 0
        self._wall_start = None

    def displacement(self, t_s):
        """Scripted dot displacement (N, 2) at time t_s."""
        cycle = int(t_s // CYCLE_S)
        t = t_s % CYCLE_S
        # contact location and shear direction change every cycle
        contact = np.array([320.0, 230.0]) + 90 * np.array([np.cos(1.3 * cycle), np.sin(1.3 * cycle)])
        shear_dir = np.array([np.cos(2.1 * cycle + 0.5), np.sin(2.1 * cycle + 0.5)])

        rel = self.base - contact
        r = np.linalg.norm(rel, axis=1, keepdims=True)
        sigma = 110.0
        envelope = np.exp(-(r ** 2) / (2 * sigma ** 2))

        press = _smoothstep(0.8, 1.6, t) * (1 - _smoothstep(4.4, 5.3, t))
        shear = _smoothstep(2.0, 3.0, t) * (1 - _smoothstep(4.0, 4.8, t))
        wobble = 0.6 * np.sin(2 * np.pi * 7 * t) * _smoothstep(3.0, 3.2, t) * (1 - _smoothstep(3.6, 3.8, t))

        radial = 6.0 * press * envelope * rel / sigma  # dots pushed outwards around the contact
        tangential = (8.0 * shear + wobble) * envelope * shear_dir
        return radial + tangential

    def _log_intensity(self, disp):
        cx = (self.base[:, 0] + disp[:, 0]).astype(np.float32)[:, None, None]
        cy = (self.base[:, 1] + disp[:, 1]).astype(np.float32)[:, None, None]
        dist = np.sqrt((self._pxf - cx) ** 2 + (self._pyf - cy) ** 2)
        darkness = np.clip(np.float32(self.dot_radius + 0.5) - dist, 0.0, 1.0)  # anti-aliased disk
        return np.log(1.0 - np.float32(0.8) * darkness)

    def is_running(self):
        return True

    def next_batch(self, duration_ms=5):
        if self.realtime:
            now = time.perf_counter()
            if self._wall_start is None:
                self._wall_start = now
            ahead = self.t_us * 1e-6 - (now - self._wall_start)
            if ahead > 0:
                time.sleep(ahead)

        chunks = []
        for _ in range(max(1, round(duration_ms * 1000 / self.step_us))):
            self.t_us += self.step_us
            log_i = self._log_intensity(self.displacement(self.t_us * 1e-6))
            diff = log_i - self.ref
            n = np.floor(np.abs(diff) / self.contrast).astype(np.int64)
            hit = n > 0
            if np.any(hit):
                self.ref[hit] += np.sign(diff[hit]) * n[hit] * self.contrast
                k = hit.sum()
                ev = np.empty(k, dtype=EVENT_DTYPE)
                ev["x"] = self.px[hit]
                ev["y"] = self.py[hit]
                ev["polarity"] = diff[hit] > 0
                ev["timestamp"] = self.t_us - self.rng.integers(0, self.step_us, k)
                chunks.append(ev)
            n_noise = self.rng.poisson(self.noise_rate_hz * self.step_us * 1e-6)
            if n_noise:
                ev = np.empty(n_noise, dtype=EVENT_DTYPE)
                ev["x"] = self.rng.integers(0, self.resolution[0], n_noise)
                ev["y"] = self.rng.integers(0, self.resolution[1], n_noise)
                ev["polarity"] = self.rng.integers(0, 2, n_noise)
                ev["timestamp"] = self.t_us - self.rng.integers(0, self.step_us, n_noise)
                chunks.append(ev)
        if not chunks:
            return None
        ev = np.concatenate(chunks)
        return ev[np.argsort(ev["timestamp"], kind="stable")]

    def close(self):
        pass
