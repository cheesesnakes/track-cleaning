"""
find_swaps.py

Batch ID-swap fixer for Andaman reef fish tracking CSVs.

Walks every *_tracking.csv under --csv-root, scores every consecutive-
frame transition within each track, and splits every candidate above
--threshold onto a fresh true_id — in place, no prompts, no side files.

Splits within the same segment are applied in descending frame order so
a later split never clobbers an earlier one's row range.

Three parallel trees (any nesting depth). For a CSV at
<csv_root>/<sub>/<video>_tracking.csv, the script looks for:

    frames:  <frames_root>/<sub>/<video>/frame*.jpg
    video:   <video_root>/<sub>/<video>.<anything>

If a swap frame isn't already on disk and the matching source video is
found, it's decoded and written into the frames folder using the same
naming convention track_cleaning.py scans — so the labeling pass has the
real frame in hand. If --frames-root or --video-root is omitted, or the
video isn't found, splits still happen; the missing frame is just skipped
by track_cleaning.py when it advances.

What a split does: the suffix of (id, true_id) starting at the swap frame
is moved onto a fresh true_id, and its assigned_species / assigned_genus /
assigned_family columns are cleared so track_cleaning.py will re-label
that segment on the next pass. The `id` column is never touched.

Usage:
    # Preview across the whole tree
    python find_swaps.py \
        --csv-root data/annotated_videos \
        --frames-root data/annotated_frames \
        --video-root /mnt/videos \
        --dry-run

    # Apply
    python find_swaps.py \
        --csv-root data/annotated_videos \
        --frames-root data/annotated_frames \
        --video-root /mnt/videos

    # CSVs only, no frame backfill
    python find_swaps.py --csv-root data/annotated_videos

Requires:
    pip install opencv-python pandas numpy
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


# ==========================================
# Frame-index helpers
# ==========================================
def build_frame_index(root_dir):
    index = {}
    if not root_dir or not os.path.isdir(root_dir):
        return index
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                index[f] = os.path.join(dirpath, f)
    return index


def locate_frame_path(index, frame_num):
    try:
        n = int(frame_num)
    except (ValueError, TypeError):
        return None
    for candidate in (f"frame{n:06d}.jpg", f"frame{n}.jpg", f"_{n:04d}.jpg"):
        if candidate in index:
            return index[candidate]
    for name, path in index.items():
        if name.endswith(f"frame{n:06d}.jpg") or name.endswith(f"frame{n}.jpg"):
            return path
    return None


def frame_on_disk(index, frame_num):
    return locate_frame_path(index, frame_num) is not None


# ==========================================
# Video index — keyed by stem, any extension
# ==========================================
def build_video_index(root_dir):
    index = {}
    if not root_dir or not os.path.isdir(root_dir):
        return index
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            stem, ext = os.path.splitext(f)
            if ext.lower() in (".mp4", ".mov", ".avi", ".mkv", ".m4v", ".mpg", ".mpeg"):
                index.setdefault(stem, os.path.join(dirpath, f))
    return index


# ==========================================
# CSV loading / normalisation
# ==========================================
def load_df(csv_path):
    df = pd.read_csv(csv_path).reset_index(drop=True)
    if "true_id" not in df.columns:
        df["true_id"] = df["id"]
    else:
        m = df["true_id"].isna()
        df.loc[m, "true_id"] = df.loc[m, "id"]
    if "review_status" not in df.columns:
        df["review_status"] = pd.Series([pd.NA] * len(df), dtype="object")
    return df


# ==========================================
# Video decoding
# ==========================================
class VideoFrameProvider:
    """
    Decode-accurate access to individual frames of a video by absolute
    frame number.

    Frames are fetched by sequential decode from the nearest prior read
    position when the target is close by (reliable on any codec, including
    long-GOP footage where a raw CAP_PROP_POS_FRAMES seek can land a few
    frames off), and by a hard seek + short resync otherwise.
    """

    def __init__(self, video_path, max_seek_resync=5):
        self.video_path = str(video_path)
        self.max_seek_resync = max_seek_resync
        self._cap = None
        self._last_pos = -1

    def open(self):
        if not os.path.isfile(self.video_path):
            raise SystemExit(f"Video not found: {self.video_path}")
        self._cap = cv2.VideoCapture(self.video_path)
        if not self._cap.isOpened():
            raise SystemExit(f"Could not open video: {self.video_path}")
        self._last_pos = -1

    def get_frame(self, frame_num):
        """Return a BGR frame (numpy array), or None if it couldn't be read."""
        frame_num = int(frame_num)
        cap = self._cap

        if self._last_pos < 0 or not (
            self._last_pos < frame_num <= self._last_pos + 500
        ):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            self._last_pos = frame_num - 1

        frame = None
        steps = 0
        max_steps = (frame_num - self._last_pos) + self.max_seek_resync
        while self._last_pos < frame_num and steps < max_steps:
            ok, frame = cap.read()
            steps += 1
            if not ok:
                return None
            self._last_pos += 1

        if frame is None or self._last_pos != frame_num:
            return None
        return frame

    def close(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None


# ==========================================
# Discontinuity detection
# ==========================================
def _box_center(row):
    return ((row["x1"] + row["x2"]) / 2.0, (row["y1"] + row["y2"]) / 2.0)


def detect_track_discontinuities(df, *, velocity_window=5, max_pairs=500):
    """
    For every consecutive-row pair within the same track `id`, compute a
    continuity score. A high score means the transition looks like a jump —
    an occlusion-triggered identity swap, a track break, or a merged
    detection — rather than a fish smoothly continuing to swim.

    Returns a DataFrame sorted by score desc, capped at `max_pairs`, with
    columns: id, prev_idx, curr_idx, prev_frame, curr_frame, frame_gap,
    pos_jump_pf, vel_angle_deg, area_ratio, speed_ratio, score,
    true_id_prev, true_id_curr.
    """
    rows = []
    for track_id, grp in df.groupby("id", sort=False):
        grp = grp.sort_values("frame")
        idxs = grp.index.tolist()
        if len(idxs) < 2:
            continue

        cxs, cys, areas, frames = [], [], [], []
        for i in idxs:
            r = df.loc[i]
            cx, cy = _box_center(r)
            cxs.append(cx)
            cys.append(cy)
            areas.append(max(1.0, float((r["x2"] - r["x1"]) * (r["y2"] - r["y1"]))))
            frames.append(int(r["frame"]))

        for k in range(1, len(idxs)):
            prev_idx, curr_idx = idxs[k - 1], idxs[k]
            f_prev, f_curr = frames[k - 1], frames[k]
            f_gap = max(1, f_curr - f_prev)
            dx = cxs[k] - cxs[k - 1]
            dy = cys[k] - cys[k - 1]
            dist = (dx * dx + dy * dy) ** 0.5
            pos_jump_pf = dist / f_gap

            lo = max(0, k - velocity_window)
            hist = []
            for j in range(lo + 1, k):
                fj = max(1, frames[j] - frames[j - 1])
                hx = (cxs[j] - cxs[j - 1]) / fj
                hy = (cys[j] - cys[j - 1]) / fj
                hist.append((hx, hy))
            if hist:
                vx_prev = sum(h[0] for h in hist) / len(hist)
                vy_prev = sum(h[1] for h in hist) / len(hist)
            else:
                vx_prev = vy_prev = 0.0

            vx_curr = dx / f_gap
            vy_curr = dy / f_gap

            n1 = (vx_prev**2 + vy_prev**2) ** 0.5
            n2 = (vx_curr**2 + vy_curr**2) ** 0.5
            if n1 > 1e-6 and n2 > 1e-6:
                cos_a = max(
                    -1.0, min(1.0, (vx_prev * vx_curr + vy_prev * vy_curr) / (n1 * n2))
                )
                vel_angle_deg = float(np.degrees(np.arccos(cos_a)))
            else:
                vel_angle_deg = 0.0

            area_ratio = areas[k] / areas[k - 1]
            speed_ratio = n2 / (n1 + 1e-6)

            jump_score = pos_jump_pf
            angle_score = vel_angle_deg / 180.0
            area_score = abs(float(np.log(max(area_ratio, 1e-6))))
            speed_score = (
                abs(float(np.log(max(speed_ratio, 1e-6)))) if n1 > 1e-3 else 0.0
            )
            gap_score = float(np.log1p(f_gap)) / 5.0

            score = (
                1 * jump_score
                + 20.0 * angle_score
                + 0.25 * area_score
                + 1.0 * speed_score
                + 5.0 * gap_score
            )

            rows.append(
                {
                    "id": int(track_id),
                    "prev_idx": int(prev_idx),
                    "curr_idx": int(curr_idx),
                    "prev_frame": int(f_prev),
                    "curr_frame": int(f_curr),
                    "frame_gap": int(f_gap),
                    "pos_jump_pf": float(pos_jump_pf),
                    "vel_angle_deg": float(vel_angle_deg),
                    "area_ratio": float(area_ratio),
                    "speed_ratio": float(speed_ratio),
                    "true_id_prev": df.at[prev_idx, "true_id"],
                    "true_id_curr": df.at[curr_idx, "true_id"],
                    "score": float(score),
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return (
        out.sort_values("score", ascending=False).head(max_pairs).reset_index(drop=True)
    )


def _unresolved_candidates(df, threshold, max_pairs):
    """Detect discontinuities, drop already-resolved ones, filter by score."""
    disc = detect_track_discontinuities(df, max_pairs=max_pairs)
    if disc.empty:
        return disc
    disc = disc[disc["true_id_prev"] == disc["true_id_curr"]].reset_index(drop=True)
    resolved = set(df.index[df["review_status"].isin(["dismissed", "split"])].tolist())
    disc = disc[~disc["curr_idx"].isin(resolved)].reset_index(drop=True)
    disc = disc[disc["score"] >= threshold].reset_index(drop=True)
    return disc


# ==========================================
# Splitting
# ==========================================
def next_true_id(df):
    ids = pd.to_numeric(df["true_id"], errors="coerce")
    if ids.notna().any():
        return int(ids.max()) + 1
    return f"t{len(df) + 1}"


def apply_split(df, track_id, old_true_id, split_frame):
    """
    Move the suffix of (track_id, old_true_id) starting at `split_frame`
    onto a fresh true_id, clearing its assigned_* labels so the labeling
    pass picks it up again.
    """
    mask = (
        (df["id"] == track_id)
        & (df["true_id"] == old_true_id)
        & (df["frame"] >= split_frame)
    )
    n = int(mask.sum())
    if n == 0:
        return 0, None
    new_id = next_true_id(df)
    df.loc[mask, "true_id"] = new_id
    for col in ("assigned_species", "assigned_genus", "assigned_family"):
        if col in df.columns:
            df.loc[mask, col] = pd.NA
    return n, new_id


# ==========================================
# Frame backfill
# ==========================================
def backfill_frames(frame_nums, frames_index, frames_dir, video_path):
    """
    Ensure every frame in `frame_nums` exists under frames_dir. Frames
    already indexed are skipped. Missing ones are decoded from video_path
    and written using the frame%06d.jpg convention, then added to
    frames_index. Returns (n_written, n_failed).
    """
    if not frames_dir or not video_path:
        return 0, 0

    missing = sorted(
        int(f)
        for f in set(int(x) for x in frame_nums)
        if not frame_on_disk(frames_index, f)
    )
    if not missing:
        return 0, 0

    if not os.path.isfile(str(video_path)):
        return 0, 0

    provider = VideoFrameProvider(str(video_path))
    try:
        provider.open()
    except SystemExit:
        return 0, 0

    n_ok = n_fail = 0
    frames_dir = Path(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    try:
        for f in missing:
            img = provider.get_frame(f)
            if img is None:
                n_fail += 1
                continue
            out = frames_dir / f"frame{f:06d}.jpg"
            cv2.imwrite(str(out), img)
            frames_index[out.name] = str(out)
            n_ok += 1
    finally:
        provider.close()
    return n_ok, n_fail


# ==========================================
# Per-CSV processing
# ==========================================
def _resolve_paths(csv_path, args):
    """Mirror the CSV's relative path under --frames-root / --video-root."""
    rel = Path(csv_path).relative_to(args.csv_root)
    video_name = Path(csv_path).name[: -len("_tracking.csv")]

    frames_dir = None
    if args.frames_root:
        frames_dir = Path(args.frames_root) / rel.parent / video_name

    video_path = args.video_index.get(video_name) if args.video_root else None

    return frames_dir, video_path


def process_one(csv_path, args):
    """
    Detect and apply splits for one CSV. Backfills any missing swap frames
    from video along the way, if a matching video was found.

    Returns (n_candidates, n_splits, n_rows_split).
    """
    df = load_df(csv_path)
    disc = _unresolved_candidates(df, args.threshold, args.max_pairs)
    if disc.empty:
        return 0, 0, 0

    frames_dir, video_path = _resolve_paths(csv_path, args)
    frames_index = build_frame_index(str(frames_dir)) if frames_dir else {}

    if frames_dir and video_path:
        all_frames = set()
        for _, item in disc.iterrows():
            all_frames.add(int(item["prev_frame"]))
            all_frames.add(int(item["curr_frame"]))
        n_ok, n_fail = backfill_frames(all_frames, frames_index, frames_dir, video_path)
        if n_ok or n_fail:
            tail = f", {n_fail} failed" if n_fail else ""
            print(f"    ⤓ backfilled {n_ok} frame(s){tail}")

    # Apply in descending frame order per (id, true_id) so a split at a
    # later frame doesn't invalidate the prefix an earlier split targets.
    disc = disc.sort_values(
        ["id", "true_id_prev", "curr_frame"], ascending=[True, True, False]
    ).reset_index(drop=True)

    n_splits = 0
    n_rows = 0
    for _, item in disc.iterrows():
        track_id = int(item["id"])
        old_true = item["true_id_prev"]
        split_frm = int(item["curr_frame"])

        if args.dry_run:
            print(
                f"    would split id={track_id} @ f{split_frm} "
                f"(score {item['score']:.1f}, "
                f"jump {item['pos_jump_pf']:.2f} px/f, "
                f"Δdir {item['vel_angle_deg']:.0f}°, "
                f"area×{item['area_ratio']:.2f})"
            )
            n_splits += 1
            continue

        n, new_true = apply_split(df, track_id, old_true, split_frm)
        if n == 0:
            continue

        rs_mask = (df["id"] == track_id) & (df["frame"] == split_frm)
        df.loc[rs_mask, "review_status"] = "split"

        n_splits += 1
        n_rows += n
        print(
            f"    ✂ id={track_id} @ f{split_frm} "
            f"→ true_id {new_true} ({n} rows, "
            f"score {item['score']:.1f})"
        )

    if not args.dry_run and n_splits:
        df.to_csv(csv_path, index=False)

    return len(disc), n_splits, n_rows


# ==========================================
# Entry point
# ==========================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Batch-split probable ID swaps across a tracking-CSV tree, "
        "in place, one pass."
    )
    p.add_argument(
        "--csv-root",
        required=True,
        help="Root of the tracking-CSV tree (e.g. .../annotated_videos).",
    )
    p.add_argument(
        "--frames-root",
        default=None,
        help="Root of the frames tree, parallel to --csv-root. "
        "For CSV <csv_root>/<sub>/<video>_tracking.csv, frames are looked "
        "for at <frames_root>/<sub>/<video>/frame*.jpg.",
    )
    p.add_argument(
        "--video-root",
        default=None,
        help="Root of the source-video tree, parallel to --csv-root. "
        "Videos are matched by filename stem; extension can be anything.",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=30.0,
        help="Minimum discontinuity score to split on (default: 20). "
        "Run with --dry-run first to eyeball the distribution.",
    )
    p.add_argument("--max-pairs", type=int, default=500)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Report candidates only; don't write.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    csv_root = Path(args.csv_root)
    if not csv_root.is_dir():
        raise SystemExit(f"CSV root not found: {csv_root}")
    if args.frames_root and not Path(args.frames_root).is_dir():
        raise SystemExit(f"Frames root not found: {args.frames_root}")
    if args.video_root and not Path(args.video_root).is_dir():
        raise SystemExit(f"Video root not found: {args.video_root}")

    args.csv_root = csv_root
    args.video_index = build_video_index(args.video_root) if args.video_root else {}

    csvs = sorted(csv_root.rglob("*_tracking.csv"))
    if not csvs:
        print("No tracking CSVs found.")
        return

    backfill_on = bool(args.frames_root and args.video_root)
    print(
        f"{len(csvs)} CSV(s) | threshold {args.threshold}"
        f" | backfill {'on' if backfill_on else 'off'}"
        + (f" | {len(args.video_index)} video(s) indexed" if args.video_root else "")
        + (" | DRY RUN" if args.dry_run else "")
    )

    total_cands = total_splits = total_rows = 0
    touched = 0

    for i, csv_path in enumerate(csvs, 1):
        name = csv_path.name[: -len("_tracking.csv")]
        try:
            n_c, n_s, n_r = process_one(csv_path, args)
        except KeyboardInterrupt:
            print("\nInterrupted.")
            break
        except Exception as exc:  # noqa: BLE001
            print(f"[{i}/{len(csvs)}] {name} — ❌ {type(exc).__name__}: {exc}")
            continue

        total_cands += n_c
        total_splits += n_s
        total_rows += n_r
        if n_s:
            touched += 1
            print(f"[{i}/{len(csvs)}] {name} — {n_s} split(s)")
        else:
            print(f"[{i}/{len(csvs)}] {name} — clean")

    print(
        f"\nDone. {total_splits} split(s) across {touched} video(s), "
        f"{total_rows} rows reassigned."
        + (" (dry run — nothing written)" if args.dry_run else "")
    )


if __name__ == "__main__":
    main()
