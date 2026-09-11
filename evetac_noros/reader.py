"""ROS-free Evetac readout.

Replaces ``ebcam-readout/track_dots_ros.py``: reads events from an iniVation camera (or an
.aedat4 recording, or a synthetic generator), applies the same background-activity noise
filter, runs the dot tracker on fixed-duration slices and emits one :class:`EvetacFrame`
per output period (default 50 Hz, like the ROS node).

Example (see read.py for the command line tool):
    from evetac_noros import EvetacReader, CameraSource
    with EvetacReader(CameraSource()) as reader:
        for frame in reader.frames():
            print(frame.timestamp, len(frame.events), frame.displacement_xy.mean(axis=0))
"""

import datetime
import os
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from .tracker import DotTracker

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CALIBRATION = os.path.join(REPO_ROOT, "calibration", "calibrations", "base_calibration.pkl")

EVENT_DTYPE = np.dtype([("timestamp", "<i8"), ("x", "<i2"), ("y", "<i2"), ("polarity", "i1")])


def _dv():
    # imported lazily so the synthetic source / tracker work without dv_processing
    import dv_processing as dv
    return dv


def list_cameras():
    """Serial numbers / models of all connected iniVation cameras."""
    return [(d.serialNumber, str(d.cameraModel)) for d in _dv().io.camera.discover()]


# ---------------------------------------------------------------------------------- sources
class CameraSource:
    """Live iniVation camera (DVXplorer / DVXplorer Mini as used in Evetac)."""

    def __init__(self, serial=None, noise_filter_ms=100, packet_interval_ms=1):
        dv = _dv()
        self.camera = dv.io.camera.open(serial) if serial else dv.io.camera.open()
        self.name = self.camera.getCameraName()
        self.resolution = tuple(self.camera.getEventResolution())  # (width, height)
        # Original: camera.deviceConfigSet(-3, 1, 1000) -> commit event packets every 1 ms
        # instead of the default 10 ms to avoid unnecessary latency.
        self.camera.setTimeInterval(datetime.timedelta(milliseconds=packet_interval_ms))
        self.noise_filter = _make_noise_filter(self.resolution, noise_filter_ms)
        self.writer = None

    def record_to(self, path):
        """Additionally write the raw (unfiltered) events to an .aedat4 file."""
        self.writer = _dv().io.MonoCameraWriter(path, self.camera)

    def is_running(self):
        return self.camera.isRunning()

    def next_batch(self):
        events = self.camera.getNextEventBatch()
        if events is None:
            time.sleep(0.00025)
            return None
        if self.writer is not None:
            self.writer.writeEvents(events)
        return _filter_to_numpy(events, self.noise_filter)

    def close(self):
        self.writer = None
        self.camera = None


class FileSource:
    """Replay an .aedat4 recording (e.g. recorded with ``--record`` or DV GUI)."""

    def __init__(self, path, realtime=True, noise_filter_ms=100):
        dv = _dv()
        self.recording = dv.io.MonoCameraRecording(path)
        if not self.recording.isEventStreamAvailable():
            raise RuntimeError(f"{path} does not contain an event stream")
        self.name = f"file:{os.path.basename(path)}"
        self.resolution = tuple(self.recording.getEventResolution())
        self.realtime = realtime
        self.noise_filter = _make_noise_filter(self.resolution, noise_filter_ms)
        self._start_wall = None
        self._start_ts = None
        self._done = False

    def is_running(self):
        return not self._done

    def next_batch(self):
        events = self.recording.getNextEventBatch()
        if events is None:
            self._done = not self.recording.isRunning()
            return None
        if self.realtime and not events.isEmpty():
            if self._start_wall is None:
                self._start_wall, self._start_ts = time.perf_counter(), events.getLowestTime()
            delay = (events.getHighestTime() - self._start_ts) * 1e-6 - (time.perf_counter() - self._start_wall)
            if delay > 0:
                time.sleep(delay)
        return _filter_to_numpy(events, self.noise_filter)

    def close(self):
        self.recording = None


def _make_noise_filter(resolution, duration_ms):
    if not duration_ms:
        return None
    dv = _dv()
    return dv.noise.BackgroundActivityNoiseFilter(resolution, datetime.timedelta(milliseconds=duration_ms))


def _filter_to_numpy(events, noise_filter):
    if noise_filter is not None:
        noise_filter.accept(events)
        events = noise_filter.generateEvents()
    if events.isEmpty():
        return np.empty(0, dtype=EVENT_DTYPE)
    return np.asarray(events.numpy()).astype(EVENT_DTYPE, copy=False)


# ----------------------------------------------------------------------------------- reader
@dataclass
class EvetacFrame:
    index: int
    timestamp: int  # sensor time [us] at the end of this frame
    events: np.ndarray  # structured array (timestamp, x, y, polarity) of this frame, noise filtered
    dots_xy: np.ndarray  # (N, 2) tracked dot centers (x, y) in pixels
    initial_dots_xy: np.ndarray  # (N, 2) reference dot centers (x, y)
    tracking_active: bool

    @property
    def displacement_xy(self):
        return self.dots_xy - self.initial_dots_xy

    def event_image(self, resolution, gain=15):
        """uint8 (height, width) image, 127 = no event, like the ROS node's EvetacMsg.image.

        Note: the ROS message stored this image transposed as (width, height); here it is (height, width).
        """
        img = np.full((resolution[1], resolution[0]), 127, dtype=np.int16)
        pos = self.events["polarity"] > 0
        np.add.at(img, (self.events["y"][pos], self.events["x"][pos]), gain)
        np.add.at(img, (self.events["y"][~pos], self.events["x"][~pos]), -gain)
        return np.clip(img, 0, 255).astype(np.uint8)


class EvetacReader:
    def __init__(self, source, calibration=DEFAULT_CALIBRATION, rate_hz=50, crop=0, track=True):
        """
        source:      CameraSource / FileSource / SyntheticSource
        calibration: dot calibration .pkl
        rate_hz:     output frame rate. As in the original node the tracker runs on slices of
                     min(1000/rate_hz, 5) ms and frames are emitted every k slices.
        crop:        if < 0, only events with x < width + crop are used for tracking (original 'crop')
        """
        self.source = source
        self.resolution = source.resolution
        requested = max(1, int(round(1000 / rate_hz)))
        self.slice_ms = int(np.clip(requested, 1, 5))
        self.slices_per_frame = int(np.clip(round(requested / self.slice_ms), 1, 100))
        self.rate_hz = 1000 / (self.slice_ms * self.slices_per_frame)
        self.slice_us = self.slice_ms * 1000

        self.tracker = DotTracker(calibration, slice_ms=self.slice_ms)
        self.tracking_active = track
        self.crop_x = self.resolution[0] + crop if crop < 0 else None

        self._buffer = []
        self._next_boundary = None
        self._frame_events = []
        self._slice_count = 0
        self._frame_index = 0
        self._pending = deque()
        # timing statistics of the tracker (like the "FAILS" counter of the ROS node)
        self.slow_slices = 0
        self.total_slices = 0

    # --------------------------------------------------------------------- api
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        self.source.close()

    def frames(self):
        """Blocking generator yielding EvetacFrames until the source ends."""
        while self.source.is_running():
            frame = self.poll()
            if frame is not None:
                yield frame
        yield from self._pending

    def poll(self):
        """Read at most one batch from the source; return the oldest finished frame or None."""
        if not self._pending:
            batch = self.source.next_batch()
            if batch is not None and len(batch):
                self._ingest(batch)
        return self._pending.popleft() if self._pending else None

    def reset_reference(self):
        """Use the current dot locations as the new reference (store + reinit, in memory only)."""
        self.tracker.initial_centers = self.tracker.centers.copy()

    # --------------------------------------------------------------- internals
    def _ingest(self, batch):
        if self._next_boundary is None:
            self._next_boundary = int(batch["timestamp"][0]) + self.slice_us
        self._buffer.append(batch)
        if int(batch["timestamp"][-1]) < self._next_boundary:
            return
        events = np.concatenate(self._buffer) if len(self._buffer) > 1 else batch
        ts = events["timestamp"]
        start = 0
        while int(ts[-1]) >= self._next_boundary:
            end = int(np.searchsorted(ts, self._next_boundary, side="left"))
            self._process_slice(events[start:end])
            start = end
            self._next_boundary += self.slice_us
        self._buffer = [events[start:]] if start < len(events) else []

    def _process_slice(self, events):
        t0 = time.perf_counter()
        if self.tracking_active and len(events):
            x, y = events["x"], events["y"]
            if self.crop_x is not None:
                keep = x < self.crop_x
                x, y = x[keep], y[keep]
            self.tracker.track(x, y)
        self.total_slices += 1
        if time.perf_counter() - t0 > self.slice_ms * 1e-3:
            self.slow_slices += 1

        self._frame_events.append(events)
        self._slice_count += 1
        if self._slice_count == self.slices_per_frame:
            ev = np.concatenate(self._frame_events)
            self._pending.append(EvetacFrame(
                index=self._frame_index,
                timestamp=self._next_boundary,
                events=ev,
                dots_xy=self.tracker.dots_xy.copy(),
                initial_dots_xy=self.tracker.initial_dots_xy.copy(),
                tracking_active=self.tracking_active,
            ))
            self._frame_index += 1
            self._frame_events = []
            self._slice_count = 0


# -------------------------------------------------------------------------------------- CLI
def make_source(args):
    if args.synthetic:
        from .synthetic import SyntheticSource
        return SyntheticSource(args.calibration, realtime=not args.fast)
    if args.file:
        return FileSource(args.file, realtime=not args.fast, noise_filter_ms=args.noise_filter_ms)
    src = CameraSource(args.serial or None, noise_filter_ms=args.noise_filter_ms)
    if getattr(args, "record", None):
        src.record_to(args.record)
    return src


def add_common_args(parser):
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--serial", default="", help="camera serial number (default: first camera found)")
    g.add_argument("--file", help="replay an .aedat4 recording instead of a live camera")
    g.add_argument("--synthetic", action="store_true", help="use simulated events (no hardware needed)")
    parser.add_argument("--calibration", default=DEFAULT_CALIBRATION, help="dot calibration .pkl")
    parser.add_argument("--rate", type=float, default=50, help="output frame rate in Hz (default 50)")
    parser.add_argument("--crop", type=int, default=0, help="if < 0: ignore events with x >= width + crop for tracking")
    parser.add_argument("--noise-filter-ms", type=int, default=100, help="background activity filter duration, 0 disables")
    parser.add_argument("--record", help="(camera only) also write raw events to this .aedat4 file")
    parser.add_argument("--fast", action="store_true", help="file/synthetic: process as fast as possible, not in real time")
