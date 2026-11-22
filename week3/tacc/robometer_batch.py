"""Run published Robometer-4B on the same 16 pictures the heads see. Inference only.

Runs on a TACC GPU node inside the Robometer environment, from the repo folder:
  python week3/tacc/robometer_batch.py --data $WORK/oopsie --limit 2

Writes one line per episode to week3/robometer_scores.jsonl. Episodes already in
that file are skipped, so a job that runs out of time can simply be submitted again.
The model is loaded once, not once per video like scripts/example_inference_local.py.

--frames 8 gives Robometer 8 evenly spaced pictures instead of 16 (Robometer was trained
on 8) and writes week3/robometer_scores_8.jsonl. The test job runs both on the same 8
episodes so week3/tacc/compare_robometer_frames.py can check that 16 is no worse.
"""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "week2"))
sys.path.insert(0, os.path.dirname(HERE))

from cache_r3m import pick_rows  # noqa: E402
from loader import load_episode  # noqa: E402
from robometer.data.dataset_types import ProgressSample, Trajectory  # noqa: E402
from robometer.evals.eval_server import compute_batch_outputs  # noqa: E402
from robometer.utils.save import load_model_from_hf  # noqa: E402
from robometer.utils.setup_utils import setup_batch_collator  # noqa: E402


def tacc_path(laptop_path, data_root):
    """The index stores laptop paths. Keep the part from labs/ on and put it under data_root."""
    path = laptop_path.replace("\\", "/")
    return os.path.join(data_root, path[path.index("labs/"):])


def score(reward_model, tokenizer, collator, exp_config, frames, task, device):
    """Same steps as compute_rewards_per_frame_local in Robometer's example script."""
    traj = Trajectory(
        frames=frames,
        frames_shape=tuple(frames.shape),
        task=task,
        id="0",
        metadata={"subsequence_length": int(frames.shape[0])},
        video_embeddings=None,
    )
    batch = collator([ProgressSample(trajectory=traj, sample_type="progress")])
    inputs = batch["progress_inputs"]
    for key, value in inputs.items():
        if hasattr(value, "to"):
            inputs[key] = value.to(device)

    loss_config = getattr(exp_config, "loss", None)
    is_discrete = (
        getattr(loss_config, "progress_loss_type", "l2").lower() == "discrete" if loss_config else False
    )
    num_bins = getattr(loss_config, "progress_discrete_bins", None) or getattr(
        exp_config.model, "progress_discrete_bins", 10
    )
    # Robometer was trained with bf16 autocast. Without it, Unsloth leaves part of the model in
    # float32 and the first layer fails with "expected mat1 and mat2 to have the same dtype".
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                         enabled=device.type == "cuda"):
        results = compute_batch_outputs(
            reward_model, tokenizer, inputs, sample_type="progress", is_discrete_mode=is_discrete, num_bins=num_bins
        )

    progress = results.get("progress_pred", [])
    success = (results.get("outputs_success") or {}).get("success_probs", [])
    progress = np.array(progress[0], dtype=np.float32) if progress and len(progress) > 0 else np.array([])
    success = np.array(success[0], dtype=np.float32) if success and len(success) > 0 else np.array([])
    return progress.tolist(), success.tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="folder that holds labs/ on TACC, e.g. $WORK/oopsie")
    parser.add_argument("--out", default=None,
                        help="default week3/robometer_scores.jsonl (16 frames) or robometer_scores_<N>.jsonl")
    parser.add_argument("--model", default="robometer/Robometer-4B")
    parser.add_argument("--limit", type=int, default=0, help="test run: N successes and N failures per task")
    parser.add_argument("--frames", type=int, default=16, help="pictures per episode given to Robometer")
    args = parser.parse_args()
    if args.out is None:
        name = "robometer_scores.jsonl" if args.frames == 16 else "robometer_scores_%d.jsonl" % args.frames
        args.out = os.path.join(REPO, "week3", name)

    with open(os.path.join(REPO, "week2", "episode_index.csv"), encoding="utf-8") as f:
        rows = pick_rows(list(csv.DictReader(f)), args.limit)

    done = set()
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            done = {json.loads(line)["episode_id"] for line in f if line.strip()}
    rows = [row for row in rows if row["episode_id"] not in done]
    print("Already scored: %d. To score now: %d." % (len(done), len(rows)))
    if not rows:
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "")
    t0 = time.time()
    exp_config, tokenizer, processor, reward_model = load_model_from_hf(model_path=args.model, device=device)
    reward_model.eval()
    collator = setup_batch_collator(processor, tokenizer, exp_config, is_eval=True)
    print("Model loaded in %.0fs" % (time.time() - t0))

    seconds = []
    with open(args.out, "a", encoding="utf-8") as out:
        for number, row in enumerate(rows, start=1):
            t1 = time.time()
            episode = load_episode(tacc_path(row["h5_path"], args.data), row["task"], args.frames)
            progress, success = score(
                reward_model, tokenizer, collator, exp_config, episode["frames"], episode["instruction"], device
            )
            seconds.append(time.time() - t1)
            out.write(json.dumps({
                "task": row["task"],
                "episode_id": row["episode_id"],
                "label": row["label"],
                "split": row["split"],
                "frame_indexes": episode["frame_indexes"],
                "progress": progress,
                "success_probs": success,
                "seconds": round(seconds[-1], 2),
            }) + "\n")
            out.flush()
            print("%3d/%d %s %s  %d progress scores  %.1fs" % (
                number, len(rows), row["task"], row["label"], len(progress), seconds[-1]))

    print()
    print("Seconds per episode: %.1f (first one includes warm-up)" % (sum(seconds) / len(seconds)))
    print("Estimate for all 400 episodes: %.0f minutes plus model loading" % (np.median(seconds) * 400 / 60))


if __name__ == "__main__":
    main()
