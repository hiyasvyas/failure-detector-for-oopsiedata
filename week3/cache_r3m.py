"""Run frozen R3M on the 16 kept pictures of every episode and save the numbers.

Reads week2/episode_index.csv and uses the week-2 loader to pick the 16 pictures.
Does not train R3M and does not change any episode file or video. Runs on the
laptop CPU; 400 episodes x 16 pictures is 6,400 pictures.

Timing test first:   py -3.12 week3/cache_r3m.py --limit 2
Full run:            py -3.12 week3/cache_r3m.py
Other frame counts:  py -3.12 week3/cache_r3m.py --frames 8     (writes r3m_cache_8.pt)
                     py -3.12 week3/cache_r3m.py --frames 32    (writes r3m_cache_32.pt)
"""

import argparse
import csv
import os
import sys
import time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))


def pick_rows(rows, limit):
    """All rows, or the first `limit` successes and `limit` failures of each task."""
    if not limit:
        return rows
    kept = []
    counts = {}
    for row in rows:
        key = (row["task"], row["label"])
        if counts.get(key, 0) < limit:
            kept.append(row)
            counts[key] = counts.get(key, 0) + 1
    return kept


def to_r3m_batch(frames):
    """(frames, H, W, 3) uint8 -> (frames, 3, 224, 224) float in 0..255.

    R3M only resizes when told the input is not 224 x 224, so the cup pictures
    (640 x 480) are resized here. The whole picture is kept instead of center
    cropping, so nothing at the edges of the table is cut off.
    """
    batch = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
    if tuple(batch.shape[-2:]) != (224, 224):
        batch = F.interpolate(batch, size=(224, 224), mode="bilinear", antialias=True, align_corners=False)
    return batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=os.path.dirname(HERE), help="repo folder that holds week2/")
    parser.add_argument("--frames", type=int, default=16, help="pictures kept per episode")
    parser.add_argument("--out", default=None, help="default r3m_cache.pt for 16 frames, r3m_cache_<N>.pt otherwise")
    parser.add_argument("--limit", type=int, default=0, help="timing test: N successes and N failures per task")
    args = parser.parse_args()
    if args.out is None:
        name = "r3m_cache.pt" if args.frames == 16 else "r3m_cache_%d.pt" % args.frames
        args.out = os.path.join(HERE, name)

    sys.path.insert(0, os.path.join(args.repo, "week2"))
    from loader import load_episode
    from r3m import load_r3m

    with open(os.path.join(args.repo, "week2", "episode_index.csv"), encoding="utf-8") as f:
        rows = pick_rows(list(csv.DictReader(f)), args.limit)

    print("Loading frozen R3M resnet50. This does not train the model.")
    model = load_r3m("resnet50")
    model.eval()

    embeddings = []
    episodes = []
    skipped = []
    read_seconds = 0.0
    r3m_seconds = 0.0
    start = time.time()
    for number, row in enumerate(rows, start=1):
        t0 = time.time()
        try:
            episode = load_episode(row["h5_path"], row["task"], args.frames)
        except RuntimeError as error:
            # One unreadable video should not throw away the other 399.
            print("Skipped %s: %s" % (row["episode_id"], error))
            skipped.append({"episode_id": row["episode_id"], "error": str(error)})
            continue
        t1 = time.time()
        with torch.no_grad():
            embedding = model(to_r3m_batch(episode["frames"])).cpu()
        t2 = time.time()
        read_seconds += t1 - t0
        r3m_seconds += t2 - t1

        embeddings.append(embedding)
        episodes.append({
            "task": row["task"],
            "episode_id": row["episode_id"],
            "label": row["label"],
            "split": row["split"],
            "session_id": row["session_id"],
            "pair_id": row["pair_id"],
            "n_frames": episode["n_frames"],
            "frame_indexes": episode["frame_indexes"],
            "picture_size": list(episode["frames"].shape[1:3]),
        })
        if number % 20 == 0 or number == len(rows):
            print("%3d/%d  %s %s  read %.1fs  r3m %.1fs  total %.0fs" % (
                number, len(rows), row["task"], row["label"], t1 - t0, t2 - t1, time.time() - start))

    saved = {
        "embedding": torch.stack(embeddings),  # (episodes, frames, 2048)
        "episodes": episodes,
        "skipped": skipped,
        "note": "Frozen R3M resnet50. %d pictures per episode, resized to 224 x 224, whole picture kept." % args.frames,
    }
    torch.save(saved, args.out)
    total = time.time() - start
    print()
    done = len(episodes)
    print("Episodes: %d done, %d skipped" % (done, len(skipped)))
    print("Reading video: %.0fs total, %.2fs per episode" % (read_seconds, read_seconds / done))
    print("R3M:           %.0fs total, %.2fs per episode" % (r3m_seconds, r3m_seconds / done))
    print("Estimate for all 400 episodes: %.0f minutes" % (total / done * 400 / 60))
    print("Saved", tuple(saved["embedding"].shape), "to", args.out)


if __name__ == "__main__":
    main()
