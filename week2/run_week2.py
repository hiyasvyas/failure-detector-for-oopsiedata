"""Watch 30 attempts and check the week-2 loader.

Writes an index of all 400 episodes, saved pictures for 30 of them, and a text report.
Does not modify the dataset.
"""

import csv
import os
import sys

import imageio.v3 as iio
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from loader import index_all, load_episode, session_split

OUT = os.path.dirname(__file__)
FRAMES = os.path.join(OUT, "frames")


def choose_oven(rows):
    chosen = []
    pairs = {}
    for row in rows:
        if row["task"] != "open_oven":
            continue
        pairs.setdefault((row["session_id"], row["pair_id"]), []).append(row)
    picked_pairs = []
    for key in sorted(pairs, key=lambda item: (item[0], item[1] if item[1] is not None else -1)):
        members = pairs[key]
        successes = [row for row in members if row["label"] == "success"]
        failures = [row for row in members if row["label"] == "failure"]
        if successes and failures:
            picked_pairs.append((successes[0], failures[0], key))
    # Four pairs from the first session, three from the second, plus one extra success.
    first = [item for item in picked_pairs if item[2][0].endswith("18-14-02")][:4]
    second = [item for item in picked_pairs if item[2][0].endswith("11-59-03")][:3]
    for success, failure, _ in first + second:
        chosen.append(success)
        chosen.append(failure)
    extra = [row for row in rows if row["task"] == "open_oven" and row["label"] == "success" and row not in chosen]
    if extra:
        chosen.append(extra[0])
    return chosen[:15], first + second


def choose_cups(rows):
    chosen = []
    by_session = {}
    for row in rows:
        if row["task"] != "stack_cups":
            continue
        by_session.setdefault(row["session_id"], {"success": [], "failure": []})
        by_session[row["session_id"]][row["label"]].append(row)
    paired = []
    for session in sorted(by_session):
        successes = by_session[session]["success"]
        failures = by_session[session]["failure"]
        if successes and failures:
            paired.append((successes[0], failures[0], session))
    for success, failure, _ in paired[:7]:
        chosen.append(success)
        chosen.append(failure)
    extra = [row for row in rows if row["task"] == "stack_cups" and row["label"] == "success" and row not in chosen]
    if extra:
        chosen.append(extra[0])
    return chosen[:15], paired[:7]


def mean_abs(a, b):
    return float(np.mean(np.abs(a.astype(np.int16) - b.astype(np.int16))))


def pair_drift(success_ep, failure_ep):
    early = np.mean([mean_abs(success_ep["frames"][i], failure_ep["frames"][i]) for i in (0, 1, 2)])
    late = np.mean([mean_abs(success_ep["frames"][i], failure_ep["frames"][i]) for i in (13, 14, 15)])
    return early, late


def save_view(episode, slot):
    os.makedirs(FRAMES, exist_ok=True)
    name = "%s_%s_%02d_%s" % (episode["task"], episode["label"], slot, episode["episode_id"][:40])
    paths = {}
    for tag, index in (("start", 0), ("mid", 8), ("end", 15)):
        path = os.path.join(FRAMES, "%s_%s.jpg" % (name, tag))
        iio.imwrite(path, episode["frames"][index])
        paths[tag] = path
    return paths


def main():
    print("Indexing episode files...")
    rows = index_all()
    split = session_split(rows)
    index_path = os.path.join(OUT, "episode_index.csv")
    with open(index_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["task", "episode_id", "label", "success", "operator", "session_id", "pair_id", "split", "camera", "video_exists", "h5_path"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "task": row["task"],
                "episode_id": row["episode_id"],
                "label": row["label"],
                "success": row["success"],
                "operator": row["operator"],
                "session_id": row["session_id"],
                "pair_id": "" if row["pair_id"] is None else row["pair_id"],
                "split": split[row["h5_path"]],
                "camera": row["camera"],
                "video_exists": row["video_exists"],
                "h5_path": row["h5_path"],
            })

    oven_watch, oven_pairs = choose_oven(rows)
    cup_watch, cup_pairs = choose_cups(rows)
    watch = oven_watch + cup_watch
    print("Watching", len(watch), "episodes")

    loaded = {}
    views = []
    for slot, row in enumerate(watch, start=1):
        print(" %02d %s %s" % (slot, row["task"], row["episode_id"]))
        episode = load_episode(row["h5_path"], row["task"])
        loaded[row["h5_path"]] = episode
        paths = save_view(episode, slot)
        views.append((slot, episode, paths))
        if episode["episode_id"] == "20260803_205510":
            expected = [0, 26, 53, 79, 106, 133, 159, 186, 212, 239, 266, 292, 319, 345, 372, 399]
            if episode["frame_indexes"] != expected:
                raise RuntimeError("Frame indexes changed for the known oven failure: %s" % episode["frame_indexes"])

    lines = []
    lines.append("Week 2 loader check")
    lines.append("Indexed episodes: %d" % len(rows))
    for task in ("open_oven", "stack_cups"):
        subset = [row for row in rows if row["task"] == task]
        labels = {}
        for row in subset:
            labels[row["label"]] = labels.get(row["label"], 0) + 1
        missing = sum(1 for row in subset if not row["video_exists"])
        sessions = sorted(set(row["session_id"] for row in subset))
        train = sum(1 for row in subset if split[row["h5_path"]] == "train")
        test = len(subset) - train
        lines.append("%s: %d episodes, labels %s, missing videos %d, sessions %d, train %d, test %d" % (
            task, len(subset), labels, missing, len(sessions), train, test))
    lines.append("Test sessions:")
    for task in ("open_oven", "stack_cups"):
        names = sorted(set(row["session_id"] for row in rows if row["task"] == task and split[row["h5_path"]] == "test"))
        lines.append("  %s: %s" % (task, ", ".join(names)))

    cup_fail = [row for row in rows if row["task"] == "stack_cups" and row["label"] == "failure"]
    pending = sum(1 for row in cup_fail if "pending human review" in row["notes"])
    name_fail = sum(1 for row in rows if row["task"] == "stack_cups" and "-fail-" in row["episode_id"])
    name_fail_and_label = sum(1 for row in rows if row["task"] == "stack_cups" and "-fail-" in row["episode_id"] and row["label"] == "failure")
    lines.append("Cup failures: %d. Notes say pending human review: %d." % (len(cup_fail), pending))
    lines.append("Cup ids containing -fail-: %d. Of those, labeled failure: %d." % (name_fail, name_fail_and_label))

    lines.append("")
    lines.append("Early vs late picture difference for matched pairs (0-255 average).")
    lines.append("Early is the first 3 of the 16 pictures. Late is the last 3.")
    for success, failure, key in oven_pairs:
        early, late = pair_drift(loaded[success["h5_path"]], loaded[failure["h5_path"]])
        lines.append("oven pair %s early %.1f late %.1f" % (key[1], early, late))
    for success, failure, session in cup_pairs:
        early, late = pair_drift(loaded[success["h5_path"]], loaded[failure["h5_path"]])
        lines.append("cups %s early %.1f late %.1f" % (session[:48], early, late))

    lines.append("")
    lines.append("Watched episodes")
    for slot, episode, _ in views:
        lines.append("%02d %s %s session=%s operator=%s frames=%s first_last=%s..%s" % (
            slot, episode["task"], episode["label"], episode["session_id"], episode["operator"],
            episode["n_frames"], episode["frame_indexes"][0], episode["frame_indexes"][-1]))

    report = os.path.join(OUT, "week2_report.txt")
    with open(report, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("Wrote", report)

    html = ["<html><body><h1>Week 2 watch, 30 attempts</h1>"]
    html.append("<p>Each row is one full attempt. Start, middle, and end are 3 of the 16 pictures the loader keeps.</p>")
    for slot, episode, paths in views:
        html.append("<h2>%02d %s %s</h2>" % (slot, episode["task"], episode["label"]))
        html.append("<p>%s<br>operator %s, session %s, %d pictures in the video</p>" % (
            episode["episode_id"], episode["operator"], episode["session_id"], episode["n_frames"]))
        html.append("<img src='frames/%s' width='280'> <img src='frames/%s' width='280'> <img src='frames/%s' width='280'>" % (
            os.path.basename(paths["start"]), os.path.basename(paths["mid"]), os.path.basename(paths["end"])))
    html.append("</body></html>")
    with open(os.path.join(OUT, "watch.html"), "w", encoding="utf-8") as f:
        f.write("\n".join(html))


if __name__ == "__main__":
    main()
