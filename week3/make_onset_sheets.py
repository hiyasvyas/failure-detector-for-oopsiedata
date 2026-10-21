"""Make one picture sheet per failed test attempt, for marking where the failure starts.

Picks 20 failed test attempts per task (40 total, fixed seed). Each sheet shows the
same 16 pictures the heads see, numbered 0 to 15. A person writes the number of the
first picture where the attempt has gone wrong into onset_labels.csv.
10 of the 40 are marked "both": both annotators label those, without looking at
each other's answer, so we can check agreement.

Run:  py -3.12 week3/make_onset_sheets.py
"""

import argparse
import csv
import os
import random
import sys

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
TILE_WIDTH = 240


def sheet_image(frames):
    """4 x 4 grid of the 16 pictures with the picture number in the corner."""
    height, width = frames.shape[1:3]
    tile_height = int(round(height * TILE_WIDTH / width))
    sheet = Image.new("RGB", (4 * TILE_WIDTH, 4 * tile_height), "white")
    draw = ImageDraw.Draw(sheet)
    for index, frame in enumerate(frames):
        tile = Image.fromarray(frame).resize((TILE_WIDTH, tile_height))
        x = (index % 4) * TILE_WIDTH
        y = (index // 4) * tile_height
        sheet.paste(tile, (x, y))
        draw.rectangle([x, y, x + 28, y + 18], fill="black")
        draw.text((x + 4, y + 3), str(index), fill="white")
    return sheet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=os.path.dirname(HERE), help="repo folder that holds week2/")
    parser.add_argument("--per-task", type=int, default=20)
    parser.add_argument("--both-per-task", type=int, default=5)
    args = parser.parse_args()

    sys.path.insert(0, os.path.join(args.repo, "week2"))
    from loader import load_episode

    with open(os.path.join(args.repo, "week2", "episode_index.csv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    out = os.path.join(HERE, "onset")
    path = os.path.join(out, "onset_labels.csv")
    if os.path.exists(path):
        raise SystemExit("Not overwriting existing labels: " + path)
    sheets = os.path.join(out, "sheets")
    os.makedirs(sheets, exist_ok=True)
    rng = random.Random(0)
    table = []
    for task in ("open_oven", "stack_cups"):
        failures = [row for row in rows if row["task"] == task and row["label"] == "failure" and row["split"] == "test"]
        chosen = rng.sample(failures, min(args.per_task, len(failures)))
        for number, row in enumerate(chosen):
            if number < args.both_per_task:
                who = "both"
            else:
                who = "annotator1" if number % 2 == 0 else "annotator2"
            episode = load_episode(row["h5_path"], task)
            # Short names; the episode id is in the label table next to the sheet name.
            name = "%s_%02d.jpg" % ("oven" if task == "open_oven" else "cups", number + 1)
            sheet_image(episode["frames"]).save(os.path.join(sheets, name), quality=90)
            table.append({
                "task": task,
                "episode_id": row["episode_id"],
                "who_labels": who,
                "sheet": "sheets/" + name,
                "onset_annotator1": "",
                "onset_annotator2": "",
                "notes": "",
            })
            print("%s %s %s" % (task, who, row["episode_id"]))

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    print("Wrote %d sheets to %s and the blank label table to %s" % (len(table), sheets, path))


if __name__ == "__main__":
    main()
