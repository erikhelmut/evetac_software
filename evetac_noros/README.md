# evetac_noros — Evetac readout & visualizer without ROS

Local, ROS-free replacement for `ebcam-readout/track_dots_ros.py` and
`ebcam-visualize/pictorial_event_processing_w_initial_point_locations.py`.

- `tracker.py` — NumPy port of `DotTrackOnnx` / `DotTrackingOnnxModelFilter` (no torch, onnx or CUDA).
  Verified against the original torch model: max deviation < 1e-6 px over thousands of slices.
- `reader.py` — `EvetacReader`: camera / `.aedat4` file / synthetic source → noise filter
  (BackgroundActivityNoiseFilter, 100 ms, as in the original) → tracker on 5 ms slices → `EvetacFrame` at 50 Hz.
- `read.py` — command line tool: print stats, save dot trajectories (`.npz`), record raw events (`.aedat4`).
- `visualize.py` — live OpenCV window. Tracking mode: events + tracked dots with magnified displacement
  arrows, plots of event rate, displacement magnitude and mean shift. Raw mode (`--raw`): events only
  (polarity or per-pixel count heatmap), plots of event rate, active pixels and events per 1 ms.
- `synthetic.py` — simulated press/shear events at the calibration dot positions, for testing without hardware.

## Setup

```bash
conda activate evetac          # python 3.12, dv-processing 2.0.4, numpy, scipy, opencv-python
cd ~/evetac_software
```

USB access to the camera as a normal user needs a udev rule (once, requires sudo):

```bash
echo 'SUBSYSTEM=="usb", ATTR{idVendor}=="152a", MODE="0666"' | sudo tee /etc/udev/rules.d/65-inivation.rules
sudo udevadm control --reload-rules   # then re-plug the sensor
```

## Usage

```bash
python -m evetac_noros.read --list                 # connected cameras (serial numbers)
python -m evetac_noros.visualize                   # live view, first camera found
python -m evetac_noros.visualize --serial DXM00123 # a specific camera
python -m evetac_noros.visualize --raw             # raw events only: no tracking, no noise filter
python -m evetac_noros.visualize --synthetic       # no hardware
python -m evetac_noros.read --save run.npz --record run.aedat4   # log + raw recording
python -m evetac_noros.visualize --file run.aedat4               # replay a recording
```

Common options: `--raw`, `--calibration path.pkl`, `--rate 50` (Hz), `--crop -N`,
`--noise-filter-ms N` (default 100, or 0 with `--raw`), `--fast` (replay without real-time pacing).
Visualizer only: `--view polarity|count`, `--window-ms 0|50|200|1000`, `--arrow-gain 4`.

Visualizer keys: `q`/Esc quit · space pause display · `m` polarity/count view · `d` accumulation window
(latest frame / 50 / 200 / 1000 ms) · `c` clear plots · `h` help.
Tracking mode only: `t` toggle tracking · `r` current dots → reference · `s` store calibration · `+`/`-` arrow gain.

`s` writes `calibration/calibrations/<loaded name>_<date>_<time>.pkl` like the ROS service
`/store_current_dot_calibration`. Pass that file via `--calibration` next time.

### Raw mode

`--raw` skips the dot tracker completely, so no calibration is needed. The noise filter is off
unless you pass `--noise-filter-ms`. The event image colors each pixel by its (log-scaled,
auto-ranged) event count, or by the dominant polarity (`m`).
The "Events per 1 ms" plot bins event timestamps at 1 ms, so vibrations up to 500 Hz are
visible even though frames arrive at 50 Hz. `read.py --raw --save x.npz` stores per-frame ON/OFF counts.

## Python API

```python
from evetac_noros import EvetacReader, CameraSource

with EvetacReader(CameraSource(), rate_hz=50) as reader:
    for frame in reader.frames():
        frame.timestamp          # sensor time [us]
        frame.events             # structured array: timestamp, x, y, polarity (this 20 ms frame)
        frame.dots_xy            # (63, 2) tracked dot centers (x, y); None with track=False
        frame.displacement_xy    # (63, 2) displacement from the reference; None with track=False
        frame.event_image(reader.resolution)   # uint8 image, 127 = no event
```

`EvetacReader(source, track=False)` gives raw frames without a tracker. For non-blocking use
(e.g. inside a control loop) call `reader.poll()` repeatedly; it returns a frame or `None`.

## Differences to the ROS version

- The camera is opened with the dv-processing 2.x API (`dv.io.camera.open`), and the 1 ms packet
  interval is set via `setTimeInterval` (was `deviceConfigSet(-3, 1, 1000)`).
- Noise filtering happens per camera batch instead of per slice; the filter only depends on
  per-pixel event history, so the output is the same.
- Coordinates are (x, y) everywhere in the API. The ROS message used (y, x) Fortran-flattened
  centers and a transposed (640×480) image.
- The tracker only evaluates event/dot pairs that can contribute (KD-tree lookup), falling back
  to the exact all-pairs evaluation if tracking ever degenerates. Results are identical.
