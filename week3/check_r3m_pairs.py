"""Do matched success/failure pairs start close and drift apart in R3M numbers?

Week 2 checked this on raw pixels. This checks it on the saved R3M vectors.
For each pair it takes 1 - cosine similarity between the success and the failure
at each of the 16 pictures. Success/success pairs from the same session are the
comparison: if they drift apart just as much, the drift is not about failing.

Run:  py -3.12 week3/check_r3m_pairs.py
"""

import os
from collections import defaultdict

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))


def pairs_for(task, episodes):
    """(first, second) index pairs: success/failure and success/success, same session."""
    groups = defaultdict(lambda: {"success": [], "failure": []})
    for index, ep in enumerate(episodes):
        if ep["task"] != task:
            continue
        # Oven attempts come in matched pairs (same start). Cups are matched by session only.
        key = (ep["session_id"], ep["pair_id"]) if task == "open_oven" else ep["session_id"]
        groups[key][ep["label"]].append(index)
    fail_pairs = []
    for group in groups.values():
        fail_pairs.extend(zip(group["success"], group["failure"]))

    by_session = defaultdict(list)
    for index, ep in enumerate(episodes):
        if ep["task"] == task and ep["label"] == "success":
            by_session[ep["session_id"]].append(index)
    success_pairs = []
    for indexes in by_session.values():
        success_pairs.extend(zip(indexes[0::2], indexes[1::2]))
    return fail_pairs, success_pairs


def drift(embedding, pairs):
    """Mean distance at each of the 16 pictures, and the share of pairs where late > early."""
    curves = torch.stack([1 - F.cosine_similarity(embedding[a], embedding[b], dim=1) for a, b in pairs])
    early = curves[:, :3].mean(dim=1)
    late = curves[:, -3:].mean(dim=1)
    return curves.mean(dim=0), float(early.mean()), float(late.mean()), float((late > early).float().mean())


def main():
    cache = torch.load(os.path.join(HERE, "r3m_cache.pt"), weights_only=False)
    raw, episodes = cache["embedding"], cache["episodes"]
    # Raw R3M vectors all point in nearly the same direction, so raw distances are tiny.
    # Subtracting each task's average vector shows what actually differs between pictures.
    centered = raw.clone()
    for task in ("open_oven", "stack_cups"):
        rows = [i for i, ep in enumerate(episodes) if ep["task"] == task]
        centered[rows] -= raw[rows].mean(dim=(0, 1))

    lines = ["R3M distance between paired attempts (1 - cosine similarity, 0 means identical)."]
    lines.append("Early = first 3 of the 16 pictures. Late = last 3.")
    for version, embedding in (("raw R3M vectors", raw), ("task average subtracted", centered)):
        lines.append("")
        lines.append("== " + version)
        for task in ("open_oven", "stack_cups"):
            fail_pairs, success_pairs = pairs_for(task, episodes)
            lines.append(task)
            for name, pairs in (("success vs failure", fail_pairs), ("success vs success", success_pairs)):
                curve, early, late, share = drift(embedding, pairs)
                lines.append("  %-18s %3d pairs  early %.3f  late %.3f  late > early in %.0f%% of pairs" % (
                    name, len(pairs), early, late, 100 * share))
                lines.append("    by picture: " + " ".join("%.3f" % float(x) for x in curve))

    text = "\n".join(lines)
    print(text)
    with open(os.path.join(HERE, "r3m_pair_check.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
