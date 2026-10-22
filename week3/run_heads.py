"""Train the small heads on the saved R3M numbers and flag where failed attempts go wrong.

Runs on the laptop CPU. Needs week3/r3m_cache.pt (made by cache_r3m.py).
If week3/label_check/ has CSVs downloaded from the label-check page, cup failures
marked "looks like a success" or "not sure" are left out.
If week3/robometer_scores.jsonl exists (from TACC), Robometer goes through the same
flagging rule as the heads.

Run:          py -3.12 week3/run_heads.py
Quick test:   py -3.12 week3/run_heads.py --seeds 1

Writes to week3/results/:
  results.txt           one table per task, the numbers for the status report
  flags.csv             the flagged frame and the 16 scores for every test attempt, head, and seed
  curves_<task>.png     average score at each frame, test successes against test failures

The heads (all except D are the same two-layer network, 128 -> 64 -> 1, on the R3M numbers
shrunk from 2,048 to 128 with PCA fit on the training frames):
  A1  progress on successes only: frame t of a success gets target t/15
  A2  A1, plus target 0 on the last frame of each failure
  B   A1, plus ranking: a success's mean score must beat a paired failure's by 0.2
  C   B, plus a success/failure loss on the mean score, successes and failures weighted equally
  D   no training: how close a frame is to the nearest frame of a training success

The flagging rule, the same for every head:
  keep the highest score so far; the rule fires once the score has been at least d below
  it for s frames in a row, and the flagged frame is the frame where that happens.
  If it never fires but the last score is below 0.5, it flags the last frame (15).
d and s are picked per head on training attempts the head did not train on (5-fold
cross-fitting), aiming for 10% false flags on successes, before any test attempt is used.
"""

import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
TASKS = ("open_oven", "stack_cups")
DROPS = (0.05, 0.10, 0.20)
STAYS = (2, 3, 4)
TARGET_FALSE_FLAGS = 0.10
LAST_FRAME = 15
TRAINED_HEADS = ("A1", "A2", "B", "C")


# ---------- the flagging rule ----------

def flag_frame(scores, d, s):
    """The frame where the rule fires, or None if it never fires."""
    peak = -float("inf")
    run = 0
    for t, x in enumerate(scores):
        peak = max(peak, x)
        if x <= peak - d:
            run += 1
            if run == s:
                return t
        else:
            run = 0
    if scores[-1] < 0.5:
        return LAST_FRAME
    return None


def flag_rate(curves, d, s):
    if len(curves) == 0:
        return 0.0
    return float(np.mean([flag_frame(c, d, s) is not None for c in curves]))


def choose_setting(success_curves, failure_curves):
    """Closest to 10% false flags on successes; ties go to the setting that flags more failures."""
    best = None
    for d in DROPS:
        for s in STAYS:
            false_flags = flag_rate(success_curves, d, s)
            key = (round(abs(false_flags - TARGET_FALSE_FLAGS), 6), -flag_rate(failure_curves, d, s))
            if best is None or key < best[0]:
                best = (key, d, s, false_flags)
    return best[1], best[2], best[3]


# ---------- the heads ----------

class Head(nn.Module):
    def __init__(self, dim=128, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden, 1))

    def forward(self, x):
        return torch.sigmoid(self.net(x)).squeeze(-1)


def pair_key(meta):
    # Oven attempts come in matched pairs from the same start. Cups only share a session.
    return (meta["session_id"], meta["pair_id"]) if meta["task"] == "open_oven" else meta["session_id"]


def ranking_pairs(metas, y):
    groups = defaultdict(lambda: ([], []))
    for i, meta in enumerate(metas):
        groups[pair_key(meta)][0 if y[i] == 1 else 1].append(i)
    winners, losers = [], []
    for successes, failures in groups.values():
        for i in successes:
            for j in failures:
                winners.append(i)
                losers.append(j)
    return torch.tensor(winners, dtype=torch.long), torch.tensor(losers, dtype=torch.long)


def fit_trained(kind, X, y, metas, seed, epochs, margin=0.2, dims=128):
    """Train one head on all frames at once. Returns a function from (n, 16, 2048) to (n, 16) scores."""
    torch.manual_seed(seed)
    frames = X.reshape(-1, X.shape[-1])
    mean = frames.mean(0)
    # Shrink the 2,048 numbers to 128 with PCA on the training frames. On all 2,048 one head
    # took over a minute on the laptop; this way it takes a few seconds.
    _, _, basis = torch.pca_lowrank(frames - mean, q=dims, center=False, niter=4)
    scale = ((frames - mean) @ basis).std(0) + 1e-6

    def features(Xn):
        return ((Xn - mean) @ basis) / scale

    Z = features(X)
    model = Head(dims)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    success = y == 1
    failure = ~success
    target = torch.linspace(0, 1, X.shape[1])
    winners, losers = ranking_pairs(metas, y) if kind in ("B", "C") else (None, None)
    weights = torch.where(success, 0.5 / success.float().mean(), 0.5 / failure.float().mean().clamp(min=1e-6))

    model.train()
    for _ in range(epochs):
        scores = model(Z)
        loss = ((scores[success] - target) ** 2).mean()
        if kind == "A2" and failure.any():
            loss = loss + (scores[failure, -1] ** 2).mean()
        if kind in ("B", "C") and len(winners):
            means = scores.mean(1)
            loss = loss + F.relu(margin - (means[winners] - means[losers])).mean()
        if kind == "C":
            means = scores.mean(1).clamp(1e-6, 1 - 1e-6)
            loss = loss + (F.binary_cross_entropy(means, y, reduction="none") * weights).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()

    def score(Xn):
        with torch.no_grad():
            return model(features(Xn))
    return score


def fit_head_d(X, y):
    """Nearest stored success frame. Scaled so a typical success frame scores 1 and the
    least typical one 0, using only successes (each compared against the other successes)."""
    successes = X[y == 1]
    mean = successes.reshape(-1, X.shape[-1]).mean(0)
    bank = F.normalize((successes - mean).reshape(-1, X.shape[-1]), dim=1)
    owner = torch.arange(len(successes)).repeat_interleave(X.shape[1])
    sims = bank @ bank.T
    sims[owner[:, None] == owner[None, :]] = -float("inf")
    own_distance = 1 - sims.max(1).values
    low, high = own_distance.median(), own_distance.max()

    def score(Xn):
        query = F.normalize((Xn - mean).reshape(-1, X.shape[-1]), dim=1)
        distance = 1 - (query @ bank.T).max(1).values
        return ((high - distance) / (high - low + 1e-9)).clamp(0, 1).reshape(Xn.shape[:2])
    return score


def make_fitter(head, seed, epochs):
    if head == "D":
        return lambda X, y, metas: fit_head_d(X, y)
    return lambda X, y, metas: fit_trained(head, X, y, metas, seed, epochs)


def crossfit(fitter, X, y, metas, folds):
    """Score every training attempt with a head trained without that attempt's fold."""
    out = torch.zeros(X.shape[:2])
    for k in sorted(set(folds)):
        hold = torch.tensor([f == k for f in folds])
        keep = ~hold
        score = fitter(X[keep], y[keep], [m for m, h in zip(metas, hold) if not h])
        out[hold] = score(X[hold])
    return out


def make_folds(metas, n_folds=5):
    keys = sorted({pair_key(m) for m in metas}, key=str)
    rng = np.random.default_rng(0)
    rng.shuffle(keys)
    fold_of = {key: i % n_folds for i, key in enumerate(keys)}
    return [fold_of[pair_key(m)] for m in metas]


# ---------- data ----------

def label_check_drops(keep_unsure):
    """Cup failures the label-check page says to leave out. Several people's CSVs are merged."""
    answers = defaultdict(list)
    for path in glob.glob(os.path.join(HERE, "label_check", "*.csv")):
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("answer"):
                    answers[row["episode_id"]].append(row["answer"])
    bad = {"looks_success"} if keep_unsure else {"looks_success", "unsure"}
    drops = {eid for eid, given in answers.items() if bad & set(given)}
    return drops, len(answers)


def load_robometer():
    path = os.path.join(HERE, "robometer_scores.jsonl")
    if not os.path.exists(path):
        return {}
    scores = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                progress = np.array(row["progress"], dtype=np.float32)
                if len(progress) == 0:
                    continue
                if len(progress) != 16:
                    # Put any other number of scores onto our 16 frames.
                    progress = np.interp(np.linspace(0, 1, 16), np.linspace(0, 1, len(progress)), progress)
                scores[row["episode_id"]] = np.clip(progress, 0, 1)
    return scores


def episode_metrics(curves, labels, d, s):
    flags = [flag_frame(c, d, s) for c in curves]
    success = [f for f, lab in zip(flags, labels) if lab == 1]
    failure = [f for f, lab in zip(flags, labels) if lab == 0]
    frames = [16 if f is None else f for f in failure]
    predicted_failure = [0 if f is None else 1 for f in flags]
    return {
        "false_flags": sum(f is not None for f in success),
        "n_success": len(success),
        "failures_flagged": sum(f is not None for f in failure),
        "n_failure": len(failure),
        "median_frame": float(np.median(frames)) if frames else float("nan"),
        "before_13": sum(f is not None and f <= 12 for f in failure),
        "episode_f1": f1_score([int(1 - lab) for lab in labels], predicted_failure, zero_division=0),
    }, flags


def baselines(Xtr, ytr, Xte, yte):
    """Episode-level only: success or failure for the whole attempt, no frame."""
    rows = []
    majority = 1 if ytr.mean() >= 0.5 else 0
    rows.append(("most common label", [majority] * len(yte)))

    mean_tr, mean_te = Xtr.mean(1).numpy(), Xte.mean(1).numpy()
    scaler = StandardScaler().fit(mean_tr)
    model = LogisticRegression(max_iter=5000, class_weight="balanced").fit(scaler.transform(mean_tr), ytr)
    rows.append(("logistic regression, mean of 16 frames", model.predict(scaler.transform(mean_te)).tolist()))

    pca = PCA(n_components=32, random_state=0).fit(Xtr.reshape(-1, Xtr.shape[-1]).numpy())
    flat_tr = pca.transform(Xtr.reshape(-1, Xtr.shape[-1]).numpy()).reshape(len(Xtr), -1)
    flat_te = pca.transform(Xte.reshape(-1, Xte.shape[-1]).numpy()).reshape(len(Xte), -1)
    scaler = StandardScaler().fit(flat_tr)
    model = LogisticRegression(max_iter=5000, class_weight="balanced", C=0.1).fit(scaler.transform(flat_tr), ytr)
    rows.append(("logistic regression, all 16 frames", model.predict(scaler.transform(flat_te)).tolist()))

    out = []
    for name, predicted in rows:
        predicted = np.array(predicted)
        out.append({
            "name": name,
            "accuracy": float((predicted == yte).mean()),
            "episode_f1": f1_score((1 - yte).astype(int), (1 - predicted).astype(int), zero_division=0),
            "failures_caught": int(((predicted == 0) & (yte == 0)).sum()),
            "false_alarms": int(((predicted == 0) & (yte == 1)).sum()),
        })
    return out


# ---------- main ----------

def run_task(task, cache, drops, robometer, seeds, epochs, flag_rows, lines):
    episodes = cache["episodes"]
    index = [i for i, ep in enumerate(episodes) if ep["task"] == task and ep["episode_id"] not in drops]
    X = cache["embedding"][index]
    metas = [episodes[i] for i in index]
    y = torch.tensor([1.0 if m["label"] == "success" else 0.0 for m in metas])
    train = torch.tensor([m["split"] == "train" for m in metas])
    test = ~train
    tr_metas = [m for m, t in zip(metas, train) if t]
    te_metas = [m for m, t in zip(metas, test) if t]
    Xtr, ytr, Xte, yte = X[train], y[train], X[test], y[test]
    folds = make_folds(tr_metas)
    n_pairs = len(ranking_pairs(tr_metas, ytr)[0])
    left_out = sum(1 for ep in episodes if ep["task"] == task and ep["episode_id"] in drops)

    lines.append("")
    lines.append("=" * 100)
    lines.append("%s   train: %d successes, %d failures   test: %d successes, %d failures" % (
        task, int(ytr.sum()), int((1 - ytr).sum()), int(yte.sum()), int((1 - yte).sum())))
    lines.append("Left out after the label check: %d.   Success/failure pairs for ranking (Heads B, C): %d." % (
        left_out, n_pairs))

    heads = [(h, seed) for h in TRAINED_HEADS for seed in range(seeds)] + [("D", 0)]
    results = defaultdict(list)
    curves = {}
    for head, seed in heads:
        fitter = make_fitter(head, seed, epochs)
        cal = crossfit(fitter, Xtr, ytr, tr_metas, folds).numpy()
        d, s, cal_false = choose_setting(cal[ytr.numpy() == 1], cal[ytr.numpy() == 0])
        test_scores = fitter(Xtr, ytr, tr_metas)(Xte).numpy()
        metrics, flags = episode_metrics(test_scores, yte.numpy(), d, s)
        metrics.update({"d": d, "s": s, "cal_false": cal_false})
        results[head].append(metrics)
        if seed == 0:
            curves[head] = test_scores
        for meta, flag, sc in zip(te_metas, flags, test_scores):
            flag_rows.append([task, head, seed, meta["episode_id"], meta["label"], d, s,
                              "" if flag is None else flag, " ".join("%.3f" % v for v in sc)])
        print("  %s %s seed %d: d=%.2f s=%d, median frame %.1f" % (task, head, seed, d, s, metrics["median_frame"]))

    if robometer:
        have = [m["episode_id"] in robometer for m in metas]
        if all(have):
            rm = np.stack([robometer[m["episode_id"]] for m in metas])
            tr_np = train.numpy()
            d, s, cal_false = choose_setting(rm[tr_np & (y.numpy() == 1)], rm[tr_np & (y.numpy() == 0)])
            metrics, flags = episode_metrics(rm[~tr_np], yte.numpy(), d, s)
            metrics.update({"d": d, "s": s, "cal_false": cal_false})
            results["Robometer"].append(metrics)
            curves["Robometer"] = rm[~tr_np]
            for meta, flag, sc in zip(te_metas, flags, rm[~tr_np]):
                flag_rows.append([task, "Robometer", 0, meta["episode_id"], meta["label"], d, s,
                                  "" if flag is None else flag, " ".join("%.3f" % v for v in sc)])
        else:
            lines.append("Robometer scores are missing for %d attempts, so Robometer is skipped." % have.count(False))

    lines.append("")
    lines.append("Per-frame heads. Counts are on the test attempts, averaged over seeds.")
    lines.append("'Median frame' is the median flagged frame on test failures (0-15), counting a failure that")
    lines.append("was never flagged as 16. 'Before frame 13' means flagged before the last 3 frames.")
    lines.append("%-10s %-22s %-14s %-17s %-17s %-16s %-10s" % (
        "head", "median frame by seed", "false flags", "failures flagged", "before frame 13", "setting d, s", "episode F1"))
    for head, runs in results.items():
        lines.append("%-10s %-22s %-14s %-17s %-17s %-16s %-10s" % (
            head,
            " / ".join("%.1f" % r["median_frame"] for r in runs),
            "%.1f of %d" % (np.mean([r["false_flags"] for r in runs]), runs[0]["n_success"]),
            "%.1f of %d" % (np.mean([r["failures_flagged"] for r in runs]), runs[0]["n_failure"]),
            "%.1f of %d" % (np.mean([r["before_13"] for r in runs]), runs[0]["n_failure"]),
            " / ".join("%.2f, %d" % (r["d"], r["s"]) for r in runs),
            "%.2f" % np.mean([r["episode_f1"] for r in runs]),
        ))

    lines.append("")
    lines.append("Episode-level baselines (whole attempt only, no frame). Episode F1 treats failure as the positive class.")
    for row in baselines(Xtr, ytr.numpy(), Xte, yte.numpy()):
        lines.append("  %-40s accuracy %.2f   episode F1 %.2f   failures caught %d of %d   successes called failures %d of %d" % (
            row["name"], row["accuracy"], row["episode_f1"], row["failures_caught"], int((1 - yte).sum()),
            row["false_alarms"], int(yte.sum())))

    lines.append("")
    lines.extend(claim_check(results))
    plot(task, curves, yte.numpy())


def claim_check(results):
    """First look at the claim. Onset times are not used here (see check_onsets.py)."""
    if not all(h in results for h in ("A1", "A2", "B", "C")):
        return []
    med = {h: [r["median_frame"] for r in results[h]] for h in ("A1", "A2", "B", "C")}
    avg = {h: float(np.mean(v)) for h, v in med.items()}
    spread = max(max(v) - min(v) for v in med.values())
    earliest_a = min(avg["A1"], avg["A2"])
    one = avg["B"] <= earliest_a - 2 and avg["B"] <= 12
    two = not (avg["C"] <= avg["B"] - 2)
    out = ["First look at the claim (flagged frames only, failure start times not used yet):"]
    out.append("  average median frame: A1 %.1f, A2 %.1f, B %.1f, C %.1f   largest spread between seeds: %.1f" % (
        avg["A1"], avg["A2"], avg["B"], avg["C"], spread))
    out.append("  1. B at least 2 frames earlier than both A1 and A2, and before frame 13: %s" % ("yes" if one else "no"))
    if one and earliest_a - avg["B"] < spread:
        out.append("     (but the gap is smaller than the spread between seeds, so it counts as a tie)")
    out.append("  2. C not 2 or more frames earlier than B: %s" % ("yes" if two else "no"))
    return out


def plot(task, curves, yte):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(curves)
    fig, axes = plt.subplots(1, len(names), figsize=(3 * len(names), 3), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, names):
        c = np.asarray(curves[name])
        ax.plot(c[yte == 1].mean(0), color="#1f77b4", label="test successes")
        ax.plot(c[yte == 0].mean(0), color="#d62728", linestyle="--", label="test failures")
        ax.set_title(name)
        ax.set_xlabel("frame")
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("average score")
    axes[0].legend(fontsize=8)
    fig.suptitle(task.replace("_", " ") + " (seed 0)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "curves_%s.png" % task), dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--keep-unsure", action="store_true", help="keep cup failures marked 'not sure'")
    args = parser.parse_args()
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    os.makedirs(OUT, exist_ok=True)
    cache = torch.load(os.path.join(HERE, "r3m_cache.pt"), weights_only=False)
    drops, checked = label_check_drops(args.keep_unsure)
    robometer = load_robometer()

    lines = ["Results from week3/run_heads.py"]
    lines.append("Cup failures checked on the label-check page: %d. Left out: %d." % (checked, len(drops)))
    lines.append("Robometer scores: %s." % ("%d attempts" % len(robometer) if robometer else "not run yet"))
    flag_rows = []
    for task in TASKS:
        run_task(task, cache, drops, robometer, args.seeds, args.epochs, flag_rows, lines)

    with open(os.path.join(OUT, "flags.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["task", "head", "seed", "episode_id", "label", "d", "s", "flagged_frame", "scores"])
        writer.writerows(flag_rows)
    text = "\n".join(lines)
    with open(os.path.join(OUT, "results.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
