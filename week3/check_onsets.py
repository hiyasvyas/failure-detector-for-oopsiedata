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
3. The claim, with the marked starts in place of the old "before the last 3 frames" cutoff.
   A flag before the marked start does not count as a catch. For every marked failure the
   delay is (flagged frame - marked start); a failure flagged too early or never flagged
   counts as caught one frame past the end (16 - marked start). The claim holds if
   Prog+Ranking's median delay is at least 2 frames shorter than both Prog-Base's and
   Progress-Zero's, and Prog+Rank+Label's is not 2 or more frames shorter than
   Prog+Ranking's. A gap smaller than the spread between training seeds counts as a tie.

Run:  py -3.12 week3/check_onsets.py
For another run:  py -3.12 week3/check_onsets.py --results results_behind   (or results_frames8, ...)
Marks are on the 16-frame sheets, so flagged frames from 8- or 32-frame runs are put on
the 16-frame scale first (frame f of N becomes f * 15 / (N - 1)).
"""

import argparse
import csv
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ONSET = os.path.join(HERE, "onset")
NAMES = {"A1": "Prog-Base", "A2": "Progress-Zero", "B": "Prog+Ranking", "C": "Prog+Rank+Label",
         "D": "Distance-Baseline", "Robometer": "Robometer"}


def delay(onset, flagged):
    """Frames from the marked start to the flag. Too early or never flagged = one past the end."""
    return flagged - onset if flagged is not None and flagged >= onset else 16 - onset


def claim_lines(by_seed):
    """by_seed[head] = list of per-seed median delays. The claim check from the midterm report."""
    if not all(h in by_seed for h in ("A1", "A2", "B", "C")):
        return ["  (needs heads A1, A2, B and C in flags.csv)"]
    avg = {h: float(np.mean(by_seed[h])) for h in ("A1", "A2", "B", "C")}
    spread = max(max(v) - min(v) for h, v in by_seed.items() if h in avg)
    best_a = min(avg["A1"], avg["A2"])
    one = avg["B"] <= best_a - 2
    two = not (avg["C"] <= avg["B"] - 2)
    out = ["  median delay (frames after the marked start, averaged over seeds): %s   largest spread between seeds: %.1f" % (
        ", ".join("%s %.1f" % (NAMES[h], avg[h]) for h in ("A1", "A2", "B", "C")), spread)]
    out.append("  1. Prog+Ranking at least 2 frames sooner than both Prog-Base and Progress-Zero: %s" % (
        "yes" if one else "no"))
    if one and best_a - avg["B"] < spread:
        out.append("     (but the gap is smaller than the spread between seeds, so it counts as a tie)")
    out.append("  2. Prog+Rank+Label not 2 or more frames sooner than Prog+Ranking: %s" % ("yes" if two else "no"))
    holds = one and two and not (best_a - avg["B"] < spread)
    out.append("  The claim %s on this task." % ("holds" if holds else "does not hold"))
    return out


def read_answers(name, folder=ONSET):
    path = os.path.join(folder, "onset_%s.csv" % name)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results", help="folder in week3/ that holds flags.csv")
    parser.add_argument("--marks", default=ONSET, help="folder with onset_annotator1.csv and onset_annotator2.csv")
    parser.add_argument("--out", default=None, help="write the report here instead of <results>/onset_check.txt")
    args = parser.parse_args()
    with open(os.path.join(ONSET, "onset_sheets.csv"), encoding="utf-8") as f:
        sheets = list(csv.DictReader(f))
    first, second = read_answers("annotator1", args.marks), read_answers("annotator2", args.marks)
    if first is None or second is None:
        missing = [n for n, a in (("onset_annotator1.csv", first), ("onset_annotator2.csv", second)) if a is None]
        raise SystemExit("Missing in %s: %s" % (args.marks, ", ".join(missing)))

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

    flags_path = os.path.join(HERE, args.results, "flags.csv")
    if os.path.exists(flags_path) and onset:
        runs = defaultdict(lambda: defaultdict(list))
        rules = set()
        with open(flags_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rules.add(row.get("rule") or "drop")
                if row["label"] == "failure" and row["episode_id"] in onset:
                    frames = int(row.get("frames") or 16)
                    flagged = None if row["flagged_frame"] == "" else int(row["flagged_frame"]) * 15 / (frames - 1)
                    runs[(row["task"], row["head"])][row["seed"]].append((onset[row["episode_id"]], flagged))
        lines.append("")
        lines.append("Flagged frame against the marked start, test failures only, averaged over seeds")
        lines.append("(%s, flagging rule '%s')." % (args.results, "/".join(sorted(rules))))
        lines.append("%-11s %-20s %-8s %-11s %-15s %-24s %-8s" % (
            "task", "head", "marked", "too early", "at or after", "frames after the start", "missed"))
        medians = defaultdict(dict)
        for (task, head), by_seed in sorted(runs.items()):
            early, late, missed, delays = [], [], [], []
            for seed, pairs in by_seed.items():
                early.append(sum(1 for o, f in pairs if f is not None and f < o))
                late.append(sum(1 for o, f in pairs if f is not None and f >= o))
                missed.append(sum(1 for o, f in pairs if f is None))
                delays.extend(f - o for o, f in pairs if f is not None and f >= o)
                medians[task].setdefault(head, []).append(float(np.median([delay(o, f) for o, f in pairs])))
            n = len(next(iter(by_seed.values())))
            lines.append("%-11s %-20s %-8d %-11.1f %-15.1f %-24s %-8.1f" % (
                task, "%s %s" % (head, NAMES.get(head, "")), n, np.mean(early), np.mean(late),
                "%.1f" % np.median(delays) if delays else "-", np.mean(missed)))
        lines.append("('frames after the start' is the median over the failures flagged at or after the start.)")
        lines.append("")
        lines.append("The claim, with the marked starts in place of the 'before the last 3 frames' cutoff.")
        lines.append("Delay = flagged frame - marked start; flagged too early or never flagged = 16 - marked start.")
        for task in sorted(medians):
            lines.append(task)
            lines.extend(claim_lines(medians[task]))
        if not usable:
            lines.append("(The marks did not pass the agreement rule, so don't use these tables for the claim.)")
    elif not os.path.exists(flags_path):
        lines.append("Run week3/run_heads.py first to compare with the heads.")

    text = "\n".join(lines)
    out = args.out or os.path.join(HERE, args.results, "onset_check.txt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
