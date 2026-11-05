"""Project figure for the midterm report: frames in, frozen R3M, head, scores, flag.

Uses one cup success and one cup failure from the same test session, and Head B's
real scores for them (seed 0, 16-frame run, week3/results/flags.csv).

Run:  py -3.12 week3/make_project_figure.py
Writes proposal/figures/project_figure.png. That folder is gitignored because the
figure shows frames from OopsieData, which is private.
"""

import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "week2"))
from loader import load_episode  # noqa: E402

SUCCESS_ID = "634562821-f15628561e"
FAILURE_ID = "0759-fail-43e5a20b02"
SHOWN = (0, 4, 8, 12, 15)
SUCCESS, FAILURE = "#2a78d6", "#eb6834"
INK, MUTED, GRID, AXIS, BAND, BOX = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7", "#f1f0ec", "#f7f6f2"


def find(rows, ending):
    return next(r for r in rows if r["episode_id"].endswith(ending))


def box(ax, x, y, w, h, title, lines, size=7):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.02",
                                facecolor=BOX, edgecolor=AXIS, linewidth=0.8, clip_on=False))
    ax.text(x + w / 2, y + h - 0.035, title, ha="center", va="top", fontsize=8.5, weight="bold", color=INK)
    ax.text(x + w / 2, y + h - 0.105, lines, ha="center", va="top", fontsize=size, color=INK, linespacing=1.25)


def arrow(fig, start, end):
    fig.patches.append(matplotlib.patches.FancyArrowPatch(
        start, end, transform=fig.transFigure, arrowstyle="-|>", mutation_scale=9, color=MUTED, linewidth=1))


def main():
    with open(os.path.join(REPO, "week2", "episode_index.csv"), encoding="utf-8") as f:
        index = list(csv.DictReader(f))
    with open(os.path.join(HERE, "results", "flags.csv"), encoding="utf-8") as f:
        flags = [r for r in csv.DictReader(f) if r["task"] == "stack_cups" and r["head"] == "B" and r["seed"] == "0"]

    episodes = []
    for ending, name, color in ((SUCCESS_ID, "Success", SUCCESS), (FAILURE_ID, "Failure", FAILURE)):
        row = find(index, ending)
        flag = find(flags, ending)
        frames = load_episode(row["h5_path"], "stack_cups")["frames"]
        scores = np.array([float(x) for x in flag["scores"].split()])
        flagged = int(flag["flagged_frame"]) if flag["flagged_frame"] else None
        episodes.append((name, color, frames, scores, flagged, float(flag["d"]), int(flag["s"])))

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "text.color": INK,
                         "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})
    fig = plt.figure(figsize=(7.2, 2.5))

    # (a) The frames the heads see, 5 of the 16 for each attempt.
    left, tile_w, gap = 0.03, 0.074, 0.004
    tile_h = tile_w * 7.2 / 2.5 * 0.75
    rows_y = (0.53, 0.53 - tile_h - 0.09)
    fig.text(0.012, 0.93, "(a) 16 frames per attempt, 5 shown", fontsize=8.5, weight="bold")
    for r, (name, color, frames, *_rest) in enumerate(episodes):
        y = rows_y[r]
        fig.text(left - 0.01, y + tile_h / 2, name, rotation=90, ha="center", va="center", fontsize=8, color=color,
                 weight="bold")
        for c, t in enumerate(SHOWN):
            ax = fig.add_axes((left + c * (tile_w + gap), y, tile_w, tile_h))
            ax.imshow(frames[t])
            ax.set_xticks([])
            ax.set_yticks([])
            for side in ax.spines.values():
                side.set_color(color)
                side.set_linewidth(1.2)
            ax.set_title("frame %d" % t if c == 0 else str(t), fontsize=6.5, color=MUTED, pad=2)

    # (b) Frozen R3M and the head.
    mid = fig.add_axes((0.445, 0.04, 0.2, 0.84))
    mid.set_xlim(0, 1)
    mid.set_ylim(0, 1)
    mid.axis("off")
    fig.text(0.445, 0.93, "(b) Encoder and head", fontsize=8.5, weight="bold")
    box(mid, 0.02, 0.72, 0.96, 0.25, "Frozen R3M", "not trained by us\n2,048 numbers per frame")
    box(mid, 0.02, 0.01, 0.96, 0.65, "Head (small network)",
        "one score (0 to 1) per frame\n\ntraining signal:\n"
        "Prog-Base: progress\nProgress-Zero: + failure ends at 0\nProg+Ranking: + ranking\n"
        "Prog+Rank+Label: + label\nDistance-Baseline: no training,\ndistance to nearest success", size=6.3)
    mid.annotate("", xy=(0.5, 0.665), xytext=(0.5, 0.715),
                 arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1, mutation_scale=9))

    # (c) Prog+Ranking's (Head B's) real scores for the two attempts and the flagging rule.
    ax = fig.add_axes((0.745, 0.17, 0.24, 0.68))
    fig.text(0.68, 0.93, "(c) Prog+Ranking scores, flag", fontsize=8.5, weight="bold")
    x = np.arange(16)
    name, color, _, scores, flagged, d, s = episodes[1]
    peak = int(np.argmax(scores[:flagged + 1]))
    ax.axvspan(flagged - s + 0.5, flagged + 0.5, color=BAND, zorder=0, linewidth=0)
    ax.plot(x[peak:], np.full(16 - peak, scores[peak]), color=MUTED, linewidth=1, linestyle=(0, (1, 1.5)),
            label="failure's highest so far")
    for name, color, _, scores, _, _, _ in episodes:
        style = "-" if name == "Success" else (0, (4, 2))
        ax.plot(x, scores, color=color, linewidth=2, linestyle=style, solid_capstyle="round", label=name.lower())
    ax.plot([flagged], [episodes[1][3][flagged]], marker="o", markersize=7, color=FAILURE,
            markeredgecolor="white", markeredgewidth=1.5, zorder=5)
    ax.annotate("flag at frame %d" % flagged, (flagged, episodes[1][3][flagged]), xytext=(5.2, 0.03),
                fontsize=6.5, color=INK, va="center", arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.text(flagged - s / 2 + 0.5, 0.43,"%d frames\nbelow" % s, ha="center", va="center", fontsize=6,
            color=MUTED)
    ax.set_xlim(0, 15.5)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.5, 1])
    ax.set_xticks([0, 5, 10, 15])
    ax.set_xlabel("frame", labelpad=1)
    ax.set_ylabel("score", labelpad=1)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.legend(loc="upper left", fontsize=6, frameon=False, handlelength=2.2, borderaxespad=0.1)

    arrow(fig, (0.418, 0.47), (0.442, 0.47))
    arrow(fig, (0.652, 0.47), (0.684, 0.47))

    out_dir = os.path.join(REPO, "proposal", "figures")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "project_figure.png")
    fig.savefig(out, dpi=300, facecolor="white")
    print("Wrote", out)


if __name__ == "__main__":
    main()
