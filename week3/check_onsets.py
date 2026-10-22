"""Check the hand-marked failure starts and compare them with each head's flagged frame.

Needs, in week3/onset/: onset_sheets.csv (from make_onset_sheets.py) and the CSVs
downloaded from mark_onsets.html, onset_annotator1.csv and onset_annotator2.csv.
Needs week3/results/flags.csv (from run_heads.py) for the second part.

1. Agreement on the 10 sheets both annotators marked. The proposal's rule: if they
   disagree by more than 2 frames on more than 3 of the 10, the hand marks are not used.
   "No clear moment" against a frame number counts as a disagreement.
2. For each head, on test failures with a marked start: how many it flagged before the
   failure started (too early), how many it flagged at or after the start, how many
   frames after the start (median), and how many it never flagged.

Run:  py -3.12 week3/check_onsets.py
"""

import csv
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ONSET = os.path.join(HERE, "onset")


def read_answers(name):
    path = os.path.join(ONSET, "onset_%s.csv" % name)
    if not os.path.exists(path):
        return None
    answers = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["no_clear_moment"] == "yes":
                answers[row["episode_id"]] = "none"
            elif row["onset"].strip() != "":
                answers[row["episode_id"]] = int(row["onset"])
    return answers


def main():
    with open(os.path.join(ONSET, "onset_sheets.csv"), encoding="utf-8") as f:
        sheets = list(csv.DictReader(f))
    first, second = read_answers("annotator1"), read_answers("annotator2")
    if first is None or second is None:
        missing = [n for n, a in (("onset_annotator1.csv", first), ("onset_annotator2.csv", second)) if a is None]
        raise SystemExit("Missing in week3/onset/: " + ", ".join(missing))

    lines = ["Hand-marked failure starts"]
    both = [row for row in sheets if row["who_labels"] == "both"]
    disagree, blank = 0, 0
    lines.append("")
    lines.append("Sheets both annotators marked:")
    for row in both:
        a, b = first.get(row["episode_id"]), second.get(row["episode_id"])
        if a is None or b is None:
            blank += 1
            verdict = "not marked by both yet"
        elif a == "none" and b == "none":
            verdict = "agree (no clear moment)"
        elif "none" in (a, b) or abs(a - b) > 2:
            disagree += 1
            verdict = "DISAGREE"
        else:
            verdict = "agree (%d frames apart)" % abs(a - b)
        lines.append("  %-10s %-60s annotator 1: %-5s annotator 2: %-5s %s" % (
            row["task"], row["episode_id"][:60], a, b, verdict))
    usable = disagree <= 3 and blank == 0
    lines.append("Disagreements: %d of %d (not marked by both: %d)." % (disagree, len(both), blank))
    if blank:
        lines.append("Finish the shared sheets before deciding whether the marks can be used.")
    elif usable:
        lines.append("3 or fewer disagreements, so the hand marks can be used.")
    else:
        lines.append("More than 3 disagreements, so by our rule the hand marks are not used. Report flagged frames only.")

    onset = {}
    for row in sheets:
        a, b = first.get(row["episode_id"]), second.get(row["episode_id"])
        given = [x for x in (a, b) if x is not None]
        if not given or "none" in given:
            continue
        onset[row["episode_id"]] = int(round(np.mean(given)))
    no_moment = sum(1 for row in sheets if "none" in (first.get(row["episode_id"]), second.get(row["episode_id"])))
    lines.append("")
    lines.append("Attempts with a marked start: %d. Marked 'no clear moment': %d. Not marked: %d." % (
        len(onset), no_moment, len(sheets) - len(onset) - no_moment))

    flags_path = os.path.join(HERE, "results", "flags.csv")
    if os.path.exists(flags_path) and onset:
        runs = defaultdict(lambda: defaultdict(list))
        with open(flags_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["label"] == "failure" and row["episode_id"] in onset:
                    runs[(row["task"], row["head"])][row["seed"]].append(
                        (onset[row["episode_id"]], None if row["flagged_frame"] == "" else int(row["flagged_frame"])))
        lines.append("")
        lines.append("Flagged frame against the marked start, test failures only, averaged over seeds.")
        lines.append("%-11s %-10s %-8s %-11s %-15s %-24s %-8s" % (
            "task", "head", "marked", "too early", "at or after", "frames after the start", "missed"))
        for (task, head), by_seed in sorted(runs.items()):
            early, late, missed, delays = [], [], [], []
            for pairs in by_seed.values():
                early.append(sum(1 for o, f in pairs if f is not None and f < o))
                late.append(sum(1 for o, f in pairs if f is not None and f >= o))
                missed.append(sum(1 for o, f in pairs if f is None))
                delays.extend(f - o for o, f in pairs if f is not None and f >= o)
            n = len(next(iter(by_seed.values())))
            lines.append("%-11s %-10s %-8d %-11.1f %-15.1f %-24s %-8.1f" % (
                task, head, n, np.mean(early), np.mean(late),
                "%.1f" % np.median(delays) if delays else "-", np.mean(missed)))
        if not usable:
            lines.append("(The marks did not pass the agreement rule, so don't use this table for the claim.)")
    elif not os.path.exists(flags_path):
        lines.append("Run week3/run_heads.py first to compare with the heads.")

    text = "\n".join(lines)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "onset_check.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
