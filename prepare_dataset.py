"""Prepare the self-contained 16-frame MVFoul dataset used by this benchmark.

The temporal sampling matches SoccerNet-MVFoul/VARS: select frames from
`[start:end]` whose local index satisfies `index % factor < 1`, where
`factor = (end-start) / (((end-start)/25) * fps)`. With the paper settings
(`start=63`, `end=87`, `fps=17`), this yields original frame indices
63, 65, 66, 68, 69, 71, 72, 74, 75, 77, 78, 80, 81, 83, 84, and 86.
Annotations are copied into the prepared dataset so downstream commands no
longer depend on the original download.
"""
import argparse
import glob
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed

REPO = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SOURCE = os.path.join(REPO, "data", "mvfouls")
DEFAULT_OUTPUT = os.path.join(REPO, "data", "mvfouls_window")
START, END, FPS = 63, 87, 17
FACTOR = (END - START) / (((END - START) / 25) * FPS)   # = 25/17 ~= 1.4706


def select_local_indices(n_window):
    """VARS selection within the [start,end] window: keep j where j % factor < 1."""
    return [j for j in range(n_window) if j % FACTOR < 1]


def copy_annotations(source, output, splits):
    """Copy each selected split's annotations into the prepared dataset."""
    copied = []
    for split in splits:
        src = os.path.join(source, split, "annotations.json")
        if not os.path.isfile(src):
            raise FileNotFoundError(f"Missing annotation file: {src}")
        dst = os.path.join(output, split, "annotations.json")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(dst)
    return copied


def process_clip(args):
    src_clip, dst_clip = args
    if os.path.exists(dst_clip):
        return ("skip", src_clip, 0, 0)
    try:
        from decord import VideoReader
        import cv2
        vr = VideoReader(src_clip)
        n = len(vr)
        end = min(END, n)
        win = vr.get_batch(list(range(START, end))).asnumpy()   # (T,H,W,3) RGB uint8
        sel = select_local_indices(win.shape[0])
        frames = win[sel]
        os.makedirs(os.path.dirname(dst_clip), exist_ok=True)
        h, w = int(frames.shape[1]), int(frames.shape[2])
        tmp = dst_clip + ".tmp.mp4"
        vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), 16.0, (w, h))
        if not vw.isOpened():
            return ("err", src_clip, "cv2.VideoWriter failed to open", 0)
        for fr in frames:
            vw.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
        vw.release()
        os.replace(tmp, dst_clip)
        return ("ok", src_clip, int(frames.shape[0]), n)
    except Exception as e:
        return ("err", src_clip, str(e)[:150], 0)


def main():
    parser = argparse.ArgumentParser(
        description="Extract the paper's 16-frame temporal windows from SoccerNet-MVFoul"
    )
    parser.add_argument("splits", nargs="*", default=["Train", "Valid", "Test"],
                        help="dataset splits to process (default: Train Valid Test)")
    parser.add_argument("--source", default=DEFAULT_SOURCE,
                        help="source SoccerNet-MVFoul directory")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help="destination for the 16-frame clips and copied annotations")
    parser.add_argument("--workers", type=int,
                        default=int(os.environ.get("NPROC", "8")),
                        help="parallel worker processes (default: NPROC or 8)")
    parser.add_argument(
        "--annotations-only",
        action="store_true",
        help="copy annotations without converting videos (enough to evaluate released rankings)",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")

    source = os.path.abspath(args.source)
    output = os.path.abspath(args.output)
    if source == output:
        parser.error("--source and --output must be different directories")

    # Validate annotations before starting the expensive video conversion.
    for split in args.splits:
        annotation_path = os.path.join(source, split, "annotations.json")
        if not os.path.isfile(annotation_path):
            raise SystemExit(f"Missing annotation file: {annotation_path}")

    if args.annotations_only:
        copied = copy_annotations(source, output, args.splits)
        print(f"Copied {len(copied)} annotation files into {output}", flush=True)
        return

    tasks = []
    for split in args.splits:
        for ad in sorted(glob.glob(os.path.join(source, split, "action_*"))):
            rel = os.path.relpath(ad, source)
            for clip in sorted(glob.glob(os.path.join(ad, "clip_*.mp4"))):
                tasks.append((clip, os.path.join(output, rel, os.path.basename(clip))))
    if not tasks:
        raise SystemExit(f"No clips found under {source} for splits: {args.splits}")
    print(f"splits={args.splits}  clips={len(tasks)}  factor={FACTOR:.4f}  "
          f"selected_local_idx(24)={select_local_indices(24)}", flush=True)
    ok = skip = err = 0
    errs = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(process_clip, t) for t in tasks]
        for i, f in enumerate(as_completed(futs)):
            r = f.result()
            if r[0] == "ok":
                ok += 1
                if ok <= 2:
                    print(f"  sample ok: {os.path.basename(r[1])} -> wrote {r[2]} frames (src had {r[3]})", flush=True)
            elif r[0] == "skip":
                skip += 1
            else:
                err += 1; errs.append(r)
            if (i + 1) % 2000 == 0:
                print(f"  {i+1}/{len(tasks)}  ok={ok} skip={skip} err={err}", flush=True)
    print(f"DONE: ok={ok} skip={skip} err={err}", flush=True)
    for e in errs[:10]:
        print("  ERR", e[1], e[2], flush=True)
    if err:
        raise SystemExit(f"Window extraction failed for {err} clips; annotations were not copied")

    copied = copy_annotations(source, output, args.splits)
    print(f"Copied {len(copied)} annotation files into {output}", flush=True)


if __name__ == "__main__":
    main()
