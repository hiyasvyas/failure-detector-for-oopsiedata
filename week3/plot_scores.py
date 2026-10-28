"""One figure for the status report: average score at each frame, test successes vs test failures.

Reads week3/results/flags.csv (made by run_heads.py). Heads A1 to C are averaged over
their 3 training runs; Head D has one. The shaded band is the last 3 frames (13 to 15),
where our claim says a flag comes too late.

Run:  py -3.12 week3/plot_scores.py
Writes week3/results/average_scores.png
For another run:  py -3.12 week3/plot_scores.py --results results_frames8   (or results_cross, ...)
"""

import argparse
import csv
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
HEADS = ("A1", "A2", "B", "C", "D")
TASKS = (("open_oven", "Oven"), ("stack_cups", "Cups"))
SUCCESS, FAILURE = "#2a78d6", "#eb6834"
INK, MUTED, GRID, AXIS, BAND = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7", "#f1f0ec"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results", help="folder in week3/ that holds flags.csv")
    args = parser.parse_args()

    curves = defaultdict(list)
    trained_on = {}
    with open(os.path.join(HERE, args.results, "flags.csv"), encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["head"] in HEADS:
                curves[(row["task"], row["head"], row["label"])].append([float(x) for x in row["scores"].split()])
                trained_on[row["task"]] = row.get("trained_on") or row["task"]
    names = dict(TASKS)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
                         "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})
    fig, axes = plt.subplots(2, len(HEADS), figsize=(13, 5.6), sharex=True, sharey=True)
    frames = np.arange(len(next(iter(curves.values()))[0]))
    last = len(frames) - 1
    for r, (task, task_name) in enumerate(TASKS):
        if trained_on.get(task, task) != task:
            task_name = "Trained on %s,\ntested on %s" % (names[trained_on[task]].lower(), task_name.lower())
        for c, head in enumerate(HEADS):
            ax = axes[r, c]
            # The last 3 of 16 frames, scaled to however many frames this run used.
            ax.axvspan(12.5 * last / 15, last + 0.5, color=BAND, zorder=0, linewidth=0)
            ax.grid(axis="y", color=GRID, linewidth=0.6)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(AXIS)
            ax.plot(frames, np.mean(curves[(task, head, "success")], axis=0),
                    color=SUCCESS, linewidth=2, solid_capstyle="round", label="test successes")
            ax.plot(frames, np.mean(curves[(task, head, "failure")], axis=0),
                    color=FAILURE, linewidth=2, linestyle=(0, (4, 2)), label="test failures")
            ax.set_xlim(0, last + 0.5)
            ax.set_ylim(0, 1)
            ax.set_yticks([0, 0.5, 1])
            if r == 0:
                ax.set_title("Head " + head, fontsize=11, color=INK)
            if c == 0:
                ax.set_ylabel(task_name + "\naverage score", color=INK)
            if r == 1:
                ax.set_xlabel("frame")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles.append(plt.Rectangle((0, 0), 1, 1, color=BAND))
    labels.append("last 3 frames")
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = os.path.join(HERE, args.results, "average_scores.png")
    fig.savefig(out, dpi=200, facecolor="#fcfcfb")
    print("Wrote", out)


if __name__ == "__main__":
    main()
