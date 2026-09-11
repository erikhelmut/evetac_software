"""Command line readout: print the Evetac signal and optionally save it.

    python -m evetac_noros.read --list                    # list connected cameras
    python -m evetac_noros.read                           # first camera, print stats
    python -m evetac_noros.read --save run.npz --record run.aedat4
    python -m evetac_noros.read --file run.aedat4         # replay a recording
    python -m evetac_noros.read --synthetic               # no hardware needed
    python -m evetac_noros.read --raw                     # events only, no tracking / noise filter
"""

import argparse

import numpy as np

from .reader import EvetacReader, add_common_args, list_cameras, make_source


def main():
    parser = argparse.ArgumentParser(description="Read out Evetac without ROS and print/save the signal.")
    add_common_args(parser)
    parser.add_argument("--list", action="store_true", help="list connected cameras and exit")
    parser.add_argument("--save", help="save per-frame event counts (and dot locations if tracking) to this .npz")
    parser.add_argument("--duration", type=float, default=0, help="stop after this many seconds of sensor time")
    args = parser.parse_args()

    if args.list:
        cams = list_cameras()
        print("\n".join(f"{s}  ({m})" for s, m in cams) if cams else "No cameras found.")
        return

    source = make_source(args)
    reader = EvetacReader(source, args.calibration, rate_hz=args.rate, crop=args.crop, track=not args.raw)
    print(f"Source: {source.name}  resolution={source.resolution}  "
          f"rate={reader.rate_hz:.1f} Hz (slice {reader.slice_ms} ms x {reader.slices_per_frame})  "
          f"tracking={'off (raw)' if args.raw else 'on'}  noise filter={args.noise_filter_ms} ms")

    log = {"timestamp": [], "dots_xy": [], "n_on": [], "n_off": []}
    t_first = None
    try:
        with reader:
            for frame in reader.frames():
                t_first = frame.timestamp if t_first is None else t_first
                n_on = int(np.count_nonzero(frame.events["polarity"]))
                n_off = len(frame.events) - n_on
                if args.save:
                    log["timestamp"].append(frame.timestamp)
                    if frame.dots_xy is not None:
                        log["dots_xy"].append(frame.dots_xy.astype(np.float32))
                    log["n_on"].append(n_on)
                    log["n_off"].append(n_off)
                if frame.index % max(1, int(reader.rate_hz)) == 0:
                    line = (f"t={(frame.timestamp - t_first) / 1e6:7.2f}s  events/frame={len(frame.events):6d} "
                            f"(on {n_on}, off {n_off})")
                    if frame.dots_xy is not None:
                        disp = np.linalg.norm(frame.displacement_xy, axis=1)
                        line += (f"  mean|disp|={disp.mean():5.2f}px  max={disp.max():5.2f}px  "
                                 f"slow slices={reader.slow_slices}/{reader.total_slices}")
                    print(line)
                if args.duration and (frame.timestamp - t_first) / 1e6 >= args.duration:
                    break
    except KeyboardInterrupt:
        pass
    finally:
        if args.save and log["timestamp"]:
            data = dict(timestamp=np.array(log["timestamp"]), n_on=np.array(log["n_on"]),
                        n_off=np.array(log["n_off"]), rate_hz=reader.rate_hz)
            if log["dots_xy"]:
                data.update(dots_xy=np.stack(log["dots_xy"]), initial_dots_xy=reader.tracker.initial_dots_xy)
            np.savez_compressed(args.save, **data)
            print("Saved", args.save)


if __name__ == "__main__":
    main()
