"""Does giving Robometer 16 pictures instead of the 8 it was trained on make it worse?

Our mentor said 16 is fine if we can back it up. The test job (robometer_test.slurm) scores
the same 8 episodes twice, with 16 and with 8 pictures. This compares the two:

  VOC on the successes: how steadily the progress score rises over a successful attempt
      (Spearman rank correlation between the score and the picture number, as in the
      Robometer paper; 1 is perfect).
  How far apart the two progress curves are, after putting the 8 scores on the 16 positions.
  The flagged frame on the failures, on the 16-frame scale, with one fixed middle setting of
      the drop rule (d = 0.10; stay down 3 of 16 pictures, which is 1 of 8).

Our rule: keep 16 if its average VOC on the successes is no more than 0.05 below 8's.
With only 4 successes this is a rough check, not a proof.

Runs anywhere with numpy, on TACC or on the laptop after copying the two files back:
  python week3/tacc/compare_robometer_frames.py
  python week3/tacc/compare_robometer_frames.py --sixteen week3/robometer_test.jsonl --eight week3/robometer_test_8.jsonl
"""

import argparse
import json
import os

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def read(path):
    rows = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                if row["progress"]:
                    rows[row["episode_id"]] = row
    return rows


def voc(progress):
    """Spearman correlation between the score and the picture number."""
    p = np.asarray(progress, dtype=float)
    if len(p) < 2 or np.all(p == p[0]):
        return float("nan")
    ranks = np.argsort(np.argsort(p)).astype(float)
    order = np.arange(len(p), dtype=float)
    return float(np.corrcoef(ranks, order)[0, 1])


def drop_flag(progress, d, s):
    """The drop rule from run_heads.py (same fallback), returning the flagged picture or None."""
    peak, run = -float("inf"), 0
    for t, x in enumerate(progress):
        peak = max(peak, x)
        run = run + 1 if x <= peak - d else 0
        if run == s:
            return t
    return len(progress) - 1 if progress[-1] < 0.5 else None


def on_16(frame, n):
    return None if frame is None else frame * 15 / (n - 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sixteen", default=os.path.join(REPO, "week3", "robometer_test.jsonl"))
    parser.add_argument("--eight", default=os.path.join(REPO, "week3", "robometer_test_8.jsonl"))
    args = parser.parse_args()
    sixteen, eight = read(args.sixteen), read(args.eight)
    both = sorted(set(sixteen) & set(eight))
    if not both:
        raise SystemExit("No episode is in both files.")

    print("%-12s %-9s %-34s %-9s %-9s %-11s %-14s %-14s" % (
        "task", "label", "episode", "VOC 16", "VOC 8", "curve gap", "flag 16 (/15)", "flag 8 (/15)"))
    vocs = {16: [], 8: []}
    gaps = []
    for eid in both:
        a, b = np.asarray(sixteen[eid]["progress"], float), np.asarray(eight[eid]["progress"], float)
        b_on_a = np.interp(np.linspace(0, 1, len(a)), np.linspace(0, 1, len(b)), b)
        gap = float(np.mean(np.abs(a - b_on_a)))
        gaps.append(gap)
        label = sixteen[eid]["label"]
        v16, v8 = voc(a), voc(b)
        if label == "success":
            vocs[16].append(v16)
            vocs[8].append(v8)
        f16 = on_16(drop_flag(a, 0.10, max(1, round(3 * (len(a) - 1) / 15))), len(a))
        f8 = on_16(drop_flag(b, 0.10, max(1, round(3 * (len(b) - 1) / 15))), len(b))
        print("%-12s %-9s %-34s %-9.2f %-9.2f %-11.3f %-14s %-14s" % (
            sixteen[eid]["task"], label, eid[-34:], v16, v8, gap,
            "-" if f16 is None else "%.1f" % f16, "-" if f8 is None else "%.1f" % f8))

    print()
    if vocs[16]:
        m16, m8 = float(np.nanmean(vocs[16])), float(np.nanmean(vocs[8]))
        print("Average VOC on the %d successes: 16 pictures %.2f, 8 pictures %.2f." % (len(vocs[16]), m16, m8))
        print("Average gap between the two curves: %.3f (scores run from 0 to 1)." % np.mean(gaps))
        if m16 >= m8 - 0.05:
            print("16 pictures are no worse than 8 by our rule, so keep giving Robometer the same 16 as the heads.")
        else:
            print("16 pictures look worse than 8 by our rule. Run the full job with --frames 8 and compare")
            print("Robometer with the 8-frame heads run (week3/results_frames8 or results_behind_frames8).")
    else:
        print("No successes in both files, so VOC can't be compared.")


if __name__ == "__main__":
    main()
