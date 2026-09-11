"""NumPy port of the Evetac dot tracker (no torch / onnx / CUDA required).

This mirrors ``utils/dot_tracking_utils.py`` -> ``DotTrackOnnx`` with the default
``DotTrackingOnnxModelFilter`` model (regularized tracking) that is used by
``ebcam-readout/track_dots_ros.py``. The math is identical, only the backend changed.

Conventions (same as the original code):
  * the calibration pickle stores ``(centers, radii)`` with centers as (x, y)
  * internally the tracker keeps ``centers`` as (y, x), exactly like ``calib_center``
    in the original implementation. Use :pyattr:`dots_xy` to get (x, y).
"""

import os
import pickle
from datetime import datetime

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

GRID_HALF = 50  # the precomputed update grid covers [-50, 50] px around each dot
NEIGHBOR_DIST = 105  # dots closer than this are neighbors for the regularizer
DOT_SCALE_FACTOR = 1.1
DOT_REGULARIZATION_FACTOR = 1.5 * 200
STEP = 200 * 0.000015
REG_STEP = 0.00000025


class DotTracker:
    def __init__(self, calibration, slice_ms=5, regularize=True):
        """
        calibration: path to a calibration .pkl (as produced by calibration/calibrate.py)
        slice_ms:    duration of the event slices fed to :meth:`track` in ms (1..5). Called
                     ``downsample_factor_internally`` in the original code; it sets the
                     minimum amount of evidence required before a dot is moved.
        """
        if not 1 <= int(slice_ms) <= 5:
            raise ValueError("slice_ms must be in [1, 5] to match the original hyperparameters")
        self.slice_ms = int(slice_ms)
        self.regularize = regularize
        self.max_dots_per_event = 3  # sparse evaluation is exact while no event is near this many dots
        self.threshold_update = 10 + 3 * (self.slice_ms - 1)
        self.calibration = calibration
        self.list_of_calibrations = [calibration]
        self._load(calibration)

    # ------------------------------------------------------------------ setup
    def _load(self, calibration, centers_xy=None):
        calib_xy, radii = pickle.load(open(calibration, "rb"))
        if centers_xy is None:
            centers_xy = calib_xy
        centers_xy = np.asarray(centers_xy, dtype=np.float64).reshape(-1, 2)

        self.initial_centers = np.ascontiguousarray(np.fliplr(centers_xy))  # (y, x)
        self.centers = self.initial_centers.copy()

        dists = cdist(centers_xy, centers_xy)
        mask = dists < NEIGHBOR_DIST
        np.fill_diagonal(mask, False)
        self.neighbor_mask = mask.astype(np.float64)
        self.neighbor_dists_sq = (mask * dists) ** 2
        n_neighbors = mask.sum(axis=1)
        if n_neighbors.max() != 8:
            print("The maximum number of neighbors is not 8 - likely there is something wrong "
                  "with the regularization in the tracker!")
        # original: 8 / num_neighbors (guarded here against isolated dots)
        self.correction = 8.0 / np.maximum(n_neighbors, 1)

        self.radius = float(np.mean(radii))
        self._precompute_grid()

    def _precompute_grid(self):
        n = GRID_HALF
        coords = np.linspace(-n, n, 2 * n + 1)
        dx, dy = np.meshgrid(coords, coords, indexing="ij")
        dist = np.sqrt(dx ** 2 + dy ** 2)

        outside = (dist > self.radius) | (dist <= self.radius - 12.5 * DOT_SCALE_FACTOR)
        dx[outside] = 0
        dy[outside] = 0
        dist[dist < 1] = 1.0
        dist[outside] = 10.0

        grid_dx = 2 * (dx - self.radius * 1.25 * dx / dist)
        grid_dy = 2 * (dy - self.radius * 1.25 * dy / dist)
        # flattened lookup tables, index = (dx + 50) * 101 + (dy + 50)
        self._grid_dx = grid_dx.ravel()
        self._grid_dy = grid_dy.ravel()
        self._grid_nonzero = ((grid_dx != 0).astype(np.int32) + (grid_dy != 0)).ravel()

    # --------------------------------------------------------------- tracking
    def track(self, events_x, events_y):
        """Update the dot locations with one slice of events. Returns centers as (y, x)."""
        if len(events_x) == 0:
            return self.centers
        cy = self.centers[:, 0]
        cx = self.centers[:, 1]
        n_dots = len(cx)
        ex = events_x.astype(np.float64)
        ey = events_y.astype(np.float64)

        # The update grid is zero outside the ring of radius self.radius, so only (event, dot)
        # pairs closer than radius + sqrt(2) (margin for the integer truncation) contribute.
        # Look those up with a KD-tree instead of evaluating all N_dots x N_events pairs.
        _, nn = cKDTree(np.stack([cx, cy], axis=1)).query(
            np.stack([ex, ey], axis=1), k=list(range(1, self.max_dots_per_event + 1)),
            distance_upper_bound=self.radius + 1.5)
        if np.all(nn[:, -1] == n_dots):
            ev_i, col = np.nonzero(nn < n_dots)
            dot_i = nn[ev_i, col]
        else:  # an event is close to many dots (tracking diverged?) -> evaluate all pairs exactly
            dot_i, ev_i = np.indices((n_dots, len(ex))).reshape(2, -1)

        # float difference truncated towards zero, like torch's .long()
        dx = np.clip((ex[ev_i] - cx[dot_i]).astype(np.int64), -GRID_HALF, GRID_HALF)
        dy = np.clip((ey[ev_i] - cy[dot_i]).astype(np.int64), -GRID_HALF, GRID_HALF)
        idx = (dx + GRID_HALF) * (2 * GRID_HALF + 1) + (dy + GRID_HALF)

        decider = np.bincount(dot_i, self._grid_nonzero[idx], n_dots) >= self.threshold_update
        update_dx = np.clip(np.bincount(dot_i, self._grid_dx[idx], n_dots), -400, 400)
        update_dy = np.clip(np.bincount(dot_i, self._grid_dy[idx], n_dots), -400, 400)

        if self.regularize:
            dx_c = cx[None, :] - cx[:, None]
            dy_c = cy[None, :] - cy[:, None]
            radi = (dx_c * self.neighbor_mask) ** 2 + (dy_c * self.neighbor_mask) ** 2 - self.neighbor_dists_sq
            center_dx = self.correction * np.sum(4 * dx_c * radi, axis=1)
            center_dy = self.correction * np.sum(4 * dy_c * radi, axis=1)
            update_dx = update_dx - DOT_REGULARIZATION_FACTOR * REG_STEP * center_dx
            update_dy = update_dy - DOT_REGULARIZATION_FACTOR * REG_STEP * center_dy

        new = np.empty_like(self.centers)
        new[:, 1] = cx - STEP * decider * update_dx
        new[:, 0] = cy - STEP * decider * update_dy
        self.centers = new
        return self.centers

    # ------------------------------------------------------------ accessors
    @property
    def dots_xy(self):
        """Current dot centers as (N, 2) array of (x, y) pixel coordinates."""
        return self.centers[:, ::-1]

    @property
    def initial_dots_xy(self):
        """Reference (calibration) dot centers as (N, 2) array of (x, y)."""
        return self.initial_centers[:, ::-1]

    @property
    def displacement_xy(self):
        return self.dots_xy - self.initial_dots_xy

    # ---------------------------------------------------------- calibration
    def store_current_calibration(self, path=None):
        """Store the current dot locations as a new calibration file.

        Default filename follows the original: <loaded calibration>_<dd_mm_YYYY>_<HH_MM_SS>.pkl
        """
        if path is None:
            base, ext = os.path.splitext(self.list_of_calibrations[0])
            path = base + "_" + datetime.now().strftime("%d_%m_%Y_%H_%M_%S") + ext
        radii = np.full((self.centers.shape[0], 1), self.radius)
        with open(path, "wb") as f:
            pickle.dump((np.ascontiguousarray(self.dots_xy), radii), f)
        self.list_of_calibrations.append(path)
        print("Stored current calibration at:", path)
        return path

    def reinitialize(self):
        """Reset the tracker to the most recently stored calibration."""
        self.calibration = self.list_of_calibrations[-1]
        self._load(self.calibration)
