"""Pixel difference between the week-2 matched pairs at the start, middle, and end.

Week 2 (week2/run_week2.py) only reported the first 3 and last 3 of the 16 pictures.
This reruns the same 14 pairs (7 oven, 7 cups) and adds the middle (pictures 6 to 9).
The number is the average absolute pixel difference on a 0 to 255 scale.

Run:  py -3.12 week3/check_pixel_pairs.py
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "week2"))

from loader import index_all, load_episode  # noqa: E402
from run_week2 import choose_cups, choose_oven, mean_abs  # noqa: E402

PARTS = {"start (0-2)": range(0, 3), "middle (6-9)": range(6, 10), "end (13-15)": range(13, 16)}


def main():
    rows = index_all()
    _, oven_pairs = choose_oven(rows)
    _, cup_pairs = choose_cups(rows)
    lines = ["Average pixel difference between matched pairs (0 to 255), same pairs as week 2."]
    for task, pairs in (("oven", oven_pairs), ("cups", cup_pairs)):
        per_part = {name: [] for name in PARTS}
        for success, failure, _ in pairs:
            a = load_episode(success["h5_path"], success["task"])["frames"]
            b = load_episode(failure["h5_path"], failure["task"])["frames"]
            for name, frames in PARTS.items():
                per_part[name].append(np.mean([mean_abs(a[i], b[i]) for i in frames]))
        lines.append("%s (%d pairs)" % (task, len(pairs)))
        for name, values in per_part.items():
            lines.append("  %-13s %.1f to %.1f   average %.1f" % (name, min(values), max(values), np.mean(values)))
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(HERE, "pixel_pair_check.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
