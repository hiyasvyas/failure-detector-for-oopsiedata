"""Copy the video behind each marking sheet, named after the sheet, for annotator 2.

Each sheet in mark_onsets.html shows 16 frames from one failed attempt's video (and a
success next to it). This copies those videos out of the dataset, named like the sheets
(oven_06.mp4 is the failure on sheet oven_06, oven_06_success.mp4 the success next to it),
and lists the time in the video of each of the 16 frames, so a frame number on the sheet
can be found in the video and the video can be paused at that frame.

Only annotator 2's sheets (theirs and the "both" ones) by default; --all for all 40.
The dataset files are only read. The output is private (OopsieData) and stays out of git:
  week3/share/onset_videos/  and  week3/share/onset_videos.zip

Run:  py -3.12 week3/make_onset_videos.py
"""

import argparse
import csv
import os
import shutil
import sys
import zipfile

import imageio.v3 as iio
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "week2"))

from loader import read_record  # noqa: E402

OUT = os.path.join(HERE, "share", "onset_videos")

README = """Videos for the failure-start marking sheets

Each sheet in mark_onsets.html is 16 frames taken from one video, spread evenly from the
first frame to the last. This folder has those videos, named after the sheets:
  oven_06.mp4          the failed attempt on sheet oven_06 (the left side of the sheet)
  oven_06_success.mp4  the success shown next to it (the right side)

frame_times.csv gives, for each sheet, the time in the failure video of frames 0 to 15.
For example, if frame 7 is at 7.3 seconds, pausing the video at 7.3 s shows sheet frame 7.

You can watch the videos to understand what went wrong, but the answer is still a frame
number from 0 to 15 on the sheet, typed into mark_onsets.html as before. On the sheets
marked "both", don't compare answers with the other annotator.

These are OopsieData videos, which are private. Don't post or upload them anywhere.
"""


def frame_times(video_path, count=16):
    """Same frame numbers as week2/loader.py's sample_16, and their times in seconds."""
    n = sum(1 for _ in iio.imiter(video_path))
    fps = float(iio.immeta(video_path)["fps"])
    wanted = [int(x) for x in np.linspace(0, n - 1, count)]
    return [round(i / fps, 1) for i in wanted], n / fps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="all 40 sheets, not only annotator 2's")
    args = parser.parse_args()

    with open(os.path.join(REPO, "week2", "episode_index.csv"), encoding="utf-8") as f:
        index = {row["episode_id"]: row for row in csv.DictReader(f)}
    with open(os.path.join(HERE, "onset", "onset_sheets.csv"), encoding="utf-8") as f:
        sheets = list(csv.DictReader(f))
    if not args.all:
        sheets = [s for s in sheets if s["who_labels"] in ("annotator2", "both")]

    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT)
    rows = []
    for s in sheets:
        name = os.path.splitext(os.path.basename(s["sheet"]))[0]
        failure = read_record(index[s["episode_id"]]["h5_path"], s["task"])
        success = read_record(index[s["success_episode_id"]]["h5_path"], s["task"])
        shutil.copy2(failure["video_path"], os.path.join(OUT, name + ".mp4"))
        shutil.copy2(success["video_path"], os.path.join(OUT, name + "_success.mp4"))
        times, length = frame_times(failure["video_path"])
        row = {"sheet": name, "who_marks": s["who_labels"], "failure_video": name + ".mp4",
               "success_video": name + "_success.mp4", "video_seconds": round(length, 1),
               "failure_episode_id": s["episode_id"], "success_episode_id": s["success_episode_id"]}
        row.update({"frame_%d" % i: t for i, t in enumerate(times)})
        rows.append(row)
        print("%-8s %-10s %s  (%.0f s)" % (name, s["who_labels"], os.path.basename(failure["video_path"]), length))

    with open(os.path.join(OUT, "frame_times.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with open(os.path.join(OUT, "README.txt"), "w", encoding="utf-8") as f:
        f.write(README)

    zip_path = OUT + ".zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for name in sorted(os.listdir(OUT)):
            z.write(os.path.join(OUT, name), os.path.join("onset_videos", name))
    print()
    print("%d sheets. Wrote %s (%.1f MB). Share it privately." % (
        len(rows), zip_path, os.path.getsize(zip_path) / 1e6))


if __name__ == "__main__":
    main()
