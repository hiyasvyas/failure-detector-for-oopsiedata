"""Train the small heads on the saved R3M numbers and flag where failed attempts go wrong.

Runs on the laptop CPU. Needs week3/r3m_cache.pt (made by cache_r3m.py).
If week3/label_check/ has CSVs downloaded from the label-check page, cup failures
marked "looks like a success" or "not sure" are left out.
If week3/robometer_scores.jsonl exists (from TACC), Robometer goes through the same
flagging rule as the heads. For an 8- or 32-frame run it reads robometer_scores_8.jsonl
or robometer_scores_32.jsonl instead (robometer_batch.py --frames N), if there is one.

Run:          py -3.12 week3/run_heads.py
Quick test:   py -3.12 week3/run_heads.py --seeds 1

Other frame counts (make the cache first with cache_r3m.py --frames N):
              py -3.12 week3/run_heads.py --frames 8          writes week3/results_frames8/
              py -3.12 week3/run_heads.py --frames 32         writes week3/results_frames32/
Train on one task, test on the other:
              py -3.12 week3/run_heads.py --cross             writes week3/results_cross/
The new flagging rule:
              py -3.12 week3/run_heads.py --rule behind       writes week3/results_behind/
(--rule, --cross and --frames can be used together, e.g. results_behind_frames8/.)

Writes to week3/results/ (or the folder above):
  results.txt           one table per task, the numbers for the status report
  flags.csv             the flagged frame and the scores for every test attempt, head, and seed
  curves_<name>.png     average score at each frame, test successes against test failures

The heads (all except D are the same two-layer network, 128 -> 64 -> 1, on the R3M numbers
shrunk from 2,048 to 128 with PCA fit on the training frames):
  A1  progress on successes only: frame t of a success gets target t/(frames - 1)
  A2  A1, plus target 0 on the last frame of each failure
  B   A1, plus ranking: a success's mean score must beat a paired failure's by 0.2
  C   B, plus a success/failure loss on the mean score, successes and failures weighted equally
  D   no training: how close a frame is to the nearest frame of a training success

The flagging rule, the same for every head (pick it with --rule):
  drop    (the original rule) keep the highest score so far; the rule fires once the score
          has been at least d below it for s frames in a row, and the flagged frame is the
          frame where that happens.
  behind  (the new rule, for scores that stop rising instead of dropping) the rule fires once
          the score has been at least d below the average training-success score at the same
          frame for s frames in a row. The average comes from the cross-fitted scores below,
          so no success is scored by a head that trained on it.
  either  whichever of the two fires first.
  With every rule: if it never fires but the last score is below 0.5, it flags the last
  frame (the fallback). The results table counts how many flags came from the fallback.
d and s are picked per head on training attempts the head did not train on (5-fold
cross-fitting), aiming for 10% false flags on successes, before any test attempt is used.
When a training success is checked, its own scores are left out of the average it is
compared with.
With 16 frames s is 2, 3, or 4. With other frame counts s is scaled so it covers the same
share of the attempt (8 frames: 1 or 2; 32 frames: 4, 6, or 8).

So runs with different frame counts can be compared, flagged frames are also reported on
the 16-frame scale (frame f of N becomes f * 15 / (N - 1)), and the claim check uses that scale.
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
TASKS = ("open_oven", "stack_cups")
DROPS = (0.05, 0.10, 0.20)
STAYS_16 = (2, 3, 4)
TARGET_FALSE_FLAGS = 0.10
TRAINED_HEADS = ("A1", "A2", "B", "C")
EARLY_16 = 12  # on the 16-frame scale, flagged at frame 12 or before = before the last 3 frames
RULES = ("drop", "behind", "either")
# The names the midterm report uses.
NAMES = {"A1": "Prog-Base", "A2": "Progress-Zero", "B": "Prog+Ranking", "C": "Prog+Rank+Label",
         "D": "Distance-Baseline", "Robometer": "Robometer"}


# ---------- the flagging rule ----------

def flag_detail(scores, d, s, rule="drop", ref=None):
    """(flagged frame, how): how is "rule", "fallback", or None when nothing is flagged.
    ref is the average training-success score at each frame (needed for behind and either)."""
    peak = -float("inf")
    run_drop = run_behind = 0
    for t, x in enumerate(scores):
        peak = max(peak, x)
        if rule in ("drop", "either"):
            run_drop = run_drop + 1 if x <= peak - d else 0
            if run_drop == s:
                return t, "rule"
        if rule in ("behind", "either"):
            run_behind = run_behind + 1 if x <= ref[t] - d else 0
            if run_behind == s:
                return t, "rule"
    if scores[-1] < 0.5:
        return len(scores) - 1, "fallback"
    return None, None


def flag_frame(scores, d, s, rule="drop", ref=None):
    """The frame where the rule fires, or None if it never fires."""
    return flag_detail(scores, d, s, rule, ref)[0]


def success_reference(success_curves):
    """Average success score at each frame, and for each success the average without itself."""
    success_curves = np.asarray(success_curves)
    total, n = success_curves.sum(0), len(success_curves)
    leave_one_out = [(total - c) / max(n - 1, 1) for c in success_curves]
    return total / max(n, 1), leave_one_out


def stays_for(frames):
    """Stay-down lengths covering the same share of the attempt as 2, 3, 4 do with 16 frames."""
    return tuple(sorted({max(1, int(round(s * (frames - 1) / 15))) for s in STAYS_16}))


def to_16(frame, frames):
    """Where frame `frame` of `frames` would fall among 16 frames (0 to 15)."""
    return frame * 15 / (frames - 1)


def flag_rate(curves, d, s, rule="drop", refs=None):
    """Share of curves flagged. refs: one reference curve per curve (or None for the drop rule)."""
    if len(curves) == 0:
        return 0.0
    refs = refs if refs is not None else [None] * len(curves)
    return float(np.mean([flag_frame(c, d, s, rule, r) is not None for c, r in zip(curves, refs)]))


def choose_setting(success_curves, failure_curves, stays, rule="drop"):
    """Closest to 10% false flags on successes; ties go to the setting that flags more failures.
    For behind and either, each success is compared with the average of the other successes."""
    ref, leave_one_out = success_reference(success_curves)
    success_refs = leave_one_out if rule != "drop" else None
    failure_refs = [ref] * len(failure_curves) if rule != "drop" else None
    best = None
    for d in DROPS:
        for s in stays:
            false_flags = flag_rate(success_curves, d, s, rule, success_refs)
            key = (round(abs(false_flags - TARGET_FALSE_FLAGS), 6),
                   -flag_rate(failure_curves, d, s, rule, failure_refs))
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
    """Train one head on all frames at once. Returns a function from (n, frames, 2048) to (n, frames) scores."""
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


def load_robometer(frames=16):
    """Robometer's progress scores for this frame count: robometer_scores.jsonl for 16 frames,
    robometer_scores_<N>.jsonl otherwise (robometer_batch.py --frames N)."""
    path = os.path.join(HERE, "robometer_scores.jsonl" if frames == 16 else "robometer_scores_%d.jsonl" % frames)
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
                if len(progress) != frames:
                    # Put any other number of scores onto our frames.
                    progress = np.interp(np.linspace(0, 1, frames), np.linspace(0, 1, len(progress)), progress)
                scores[row["episode_id"]] = np.clip(progress, 0, 1)
    return scores


def select(cache, task, split, drops):
    """Embeddings, labels (1 = success), and episode info for one task and split."""
    episodes = cache["episodes"]
    index = [i for i, ep in enumerate(episodes)
             if ep["task"] == task and ep["split"] == split and ep["episode_id"] not in drops]
    metas = [episodes[i] for i in index]
    y = torch.tensor([1.0 if m["label"] == "success" else 0.0 for m in metas])
    return cache["embedding"][index], y, metas


def episode_metrics(curves, labels, d, s, rule="drop", ref=None):
    frames = len(curves[0])
    details = [flag_detail(c, d, s, rule, ref) for c in curves]
    flags = [f for f, _ in details]
    success = [f for f, lab in zip(flags, labels) if lab == 1]
    failure = [f for f, lab in zip(flags, labels) if lab == 0]
    predicted_failure = [0 if f is None else 1 for f in flags]
    return {
        "false_flags": sum(f is not None for f in success),
        "n_success": len(success),
        "failures_flagged": sum(f is not None for f in failure),
        "fallback": sum(1 for (f, how), lab in zip(details, labels) if lab == 0 and how == "fallback"),
        "n_failure": len(failure),
        # A failure that is never flagged counts as one past the last frame.
        "median_frame": float(np.median([frames if f is None else f for f in failure])) if failure else float("nan"),
        "median_16": float(np.median([16 if f is None else to_16(f, frames) for f in failure])) if failure else float("nan"),
        "early": sum(f is not None and to_16(f, frames) <= EARLY_16 + 1e-9 for f in failure),
        "episode_f1": f1_score([int(1 - lab) for lab in labels], predicted_failure, zero_division=0),
    }, details


def baselines(Xtr, ytr, Xte, yte):
    """Episode-level only: success or failure for the whole attempt, no frame."""
    frames = Xtr.shape[1]
    rows = []
    majority = 1 if ytr.mean() >= 0.5 else 0
    rows.append(("most common label", [majority] * len(yte)))

    mean_tr, mean_te = Xtr.mean(1).numpy(), Xte.mean(1).numpy()
    scaler = StandardScaler().fit(mean_tr)
    model = LogisticRegression(max_iter=5000, class_weight="balanced").fit(scaler.transform(mean_tr), ytr)
    rows.append(("logistic regression, mean of %d frames" % frames, model.predict(scaler.transform(mean_te)).tolist()))

    pca = PCA(n_components=32, random_state=0).fit(Xtr.reshape(-1, Xtr.shape[-1]).numpy())
    flat_tr = pca.transform(Xtr.reshape(-1, Xtr.shape[-1]).numpy()).reshape(len(Xtr), -1)
    flat_te = pca.transform(Xte.reshape(-1, Xte.shape[-1]).numpy()).reshape(len(Xte), -1)
    scaler = StandardScaler().fit(flat_tr)
    model = LogisticRegression(max_iter=5000, class_weight="balanced", C=0.1).fit(scaler.transform(flat_tr), ytr)
    rows.append(("logistic regression, all %d frames" % frames, model.predict(scaler.transform(flat_te)).tolist()))

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

def flag_row(test_task, train_task, frames, head, seed, meta, d, s, detail, scores, rule, ref):
    flag, how = detail
    return [test_task, train_task, frames, head, seed, meta["episode_id"], meta["label"], d, s,
            "" if flag is None else flag, " ".join("%.3f" % v for v in scores), rule, how or "",
            "" if rule == "drop" else " ".join("%.3f" % v for v in ref)]


def evaluate(train_task, test_task, cache, drops, robometer, seeds, epochs, out_dir, flag_rows, lines, rule="drop"):
    """Train on train_task's training split and test on test_task's test split.
    For the normal run the two tasks are the same."""
    Xtr, ytr, tr_metas = select(cache, train_task, "train", drops)
    Xte, yte, te_metas = select(cache, test_task, "test", drops)
    frames = Xtr.shape[1]
    stays = stays_for(frames)
    folds = make_folds(tr_metas)
    n_pairs = len(ranking_pairs(tr_metas, ytr)[0])
    left_out = sum(1 for ep in cache["episodes"] if ep["task"] in (train_task, test_task) and ep["episode_id"] in drops)
    name = test_task if train_task == test_task else "%s_to_%s" % (train_task, test_task)

    lines.append("")
    lines.append("=" * 100)
    if train_task == test_task:
        lines.append("%s   train: %d successes, %d failures   test: %d successes, %d failures" % (
            test_task, int(ytr.sum()), int((1 - ytr).sum()), int(yte.sum()), int((1 - yte).sum())))
    else:
        lines.append("Trained on %s (%d successes, %d failures), tested on %s (%d successes, %d failures)" % (
            train_task, int(ytr.sum()), int((1 - ytr).sum()), test_task, int(yte.sum()), int((1 - yte).sum())))
        lines.append("The flag setting is picked on %s only. Nothing from %s is used before testing." % (
            train_task, test_task))
    lines.append("Left out after the label check: %d.   Success/failure pairs for ranking (Heads B, C): %d." % (
        left_out, n_pairs))

    heads = [(h, seed) for h in TRAINED_HEADS for seed in range(seeds)] + [("D", 0)]
    results = defaultdict(list)
    curves = {}
    for head, seed in heads:
        fitter = make_fitter(head, seed, epochs)
        cal = crossfit(fitter, Xtr, ytr, tr_metas, folds).numpy()
        d, s, cal_false = choose_setting(cal[ytr.numpy() == 1], cal[ytr.numpy() == 0], stays, rule)
        # The behind rule compares test attempts with the average held-out training-success score.
        ref = success_reference(cal[ytr.numpy() == 1])[0]
        test_scores = fitter(Xtr, ytr, tr_metas)(Xte).numpy()
        metrics, details = episode_metrics(test_scores, yte.numpy(), d, s, rule, ref)
        metrics.update({"d": d, "s": s, "cal_false": cal_false})
        results[head].append(metrics)
        if seed == 0:
            curves[head] = test_scores
        for meta, detail, sc in zip(te_metas, details, test_scores):
            flag_rows.append(flag_row(test_task, train_task, frames, head, seed, meta, d, s, detail, sc, rule, ref))
        print("  %s %s seed %d: d=%.2f s=%d, median frame %.1f" % (name, head, seed, d, s, metrics["median_frame"]))

    if robometer:
        have = [m["episode_id"] in robometer for m in tr_metas + te_metas]
        if all(have):
            rm_tr = np.stack([robometer[m["episode_id"]] for m in tr_metas])
            rm_te = np.stack([robometer[m["episode_id"]] for m in te_metas])
            d, s, cal_false = choose_setting(rm_tr[ytr.numpy() == 1], rm_tr[ytr.numpy() == 0], stays, rule)
            ref = success_reference(rm_tr[ytr.numpy() == 1])[0]
            metrics, details = episode_metrics(rm_te, yte.numpy(), d, s, rule, ref)
            metrics.update({"d": d, "s": s, "cal_false": cal_false})
            results["Robometer"].append(metrics)
            curves["Robometer"] = rm_te
            for meta, detail, sc in zip(te_metas, details, rm_te):
                flag_rows.append(flag_row(test_task, train_task, frames, "Robometer", 0, meta, d, s, detail, sc, rule, ref))
        else:
            lines.append("Robometer scores are missing for %d attempts, so Robometer is skipped." % have.count(False))

    lines.append("")
    lines.append("Per-frame heads, flagging rule '%s'. Counts are on the test attempts, averaged over seeds." % rule)
    lines.append("'Median frame' is the median flagged frame on test failures (0-%d), counting a failure that" % (frames - 1))
    lines.append("was never flagged as %d. 'Flagged early' means flagged before the last 3 of 16 frames" % frames)
    lines.append("(in the first 80% of the attempt). 'From fallback' counts the flagged failures that only")
    lines.append("the fallback caught (no drop, last score below 0.5).")
    if frames != 16:
        lines.append("'Median, 16-frame scale' puts the median frame where it would fall among 16 frames (0-15).")
    header = ["head", "median frame by seed"]
    if frames != 16:
        header.append("median, 16-frame scale")
    header += ["false flags", "failures flagged", "from fallback", "flagged early", "setting d, s", "episode F1"]
    widths = [22, 22] + ([24] if frames != 16 else []) + [14, 17, 14, 17, 16, 10]
    lines.append(" ".join("%-*s" % (w, h) for w, h in zip(widths, header)))
    for head, runs in results.items():
        label = head if head == "Robometer" else "%s %s" % (head, NAMES[head])
        cells = [label, " / ".join("%.1f" % r["median_frame"] for r in runs)]
        if frames != 16:
            cells.append(" / ".join("%.1f" % r["median_16"] for r in runs))
        cells += [
            "%.1f of %d" % (np.mean([r["false_flags"] for r in runs]), runs[0]["n_success"]),
            "%.1f of %d" % (np.mean([r["failures_flagged"] for r in runs]), runs[0]["n_failure"]),
            "%.1f" % np.mean([r["fallback"] for r in runs]),
            "%.1f of %d" % (np.mean([r["early"] for r in runs]), runs[0]["n_failure"]),
            " / ".join("%.2f, %d" % (r["d"], r["s"]) for r in runs),
            "%.2f" % np.mean([r["episode_f1"] for r in runs]),
        ]
        lines.append(" ".join("%-*s" % (w, c) for w, c in zip(widths, cells)))

    lines.append("")
    lines.append("Episode-level baselines (whole attempt only, no frame). Episode F1 treats failure as the positive class.")
    for row in baselines(Xtr, ytr.numpy(), Xte, yte.numpy()):
        lines.append("  %-40s accuracy %.2f   episode F1 %.2f   failures caught %d of %d   successes called failures %d of %d" % (
            row["name"], row["accuracy"], row["episode_f1"], row["failures_caught"], int((1 - yte).sum()),
            row["false_alarms"], int(yte.sum())))

    lines.append("")
    lines.extend(claim_check(results))
    plot(name, curves, yte.numpy(), out_dir)


def claim_check(results):
    """First look at the claim, on the 16-frame scale. Onset times are not used here (see check_onsets.py)."""
    if not all(h in results for h in ("A1", "A2", "B", "C")):
        return []
    med = {h: [r["median_16"] for r in results[h]] for h in ("A1", "A2", "B", "C")}
    avg = {h: float(np.mean(v)) for h, v in med.items()}
    spread = max(max(v) - min(v) for v in med.values())
    earliest_a = min(avg["A1"], avg["A2"])
    one = avg["B"] <= earliest_a - 2 and avg["B"] <= EARLY_16
    two = not (avg["C"] <= avg["B"] - 2)
    out = ["First look at the claim (flagged frames on the 16-frame scale, failure start times not used;"]
    out.append("the check against the hand-marked failure starts is in check_onsets.py):")
    out.append("  average median frame: A1 %.1f, A2 %.1f, B %.1f, C %.1f   largest spread between seeds: %.1f" % (
        avg["A1"], avg["A2"], avg["B"], avg["C"], spread))
    out.append("  1. B at least 2 frames earlier than both A1 and A2, and before frame 13 (the old cutoff): %s" % (
        "yes" if one else "no"))
    if one and earliest_a - avg["B"] < spread:
        out.append("     (but the gap is smaller than the spread between seeds, so it counts as a tie)")
    out.append("  2. C not 2 or more frames earlier than B: %s" % ("yes" if two else "no"))
    return out


def plot(name, curves, yte, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(curves)
    fig, axes = plt.subplots(1, len(names), figsize=(3 * len(names), 3), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, head in zip(axes, names):
        c = np.asarray(curves[head])
        ax.plot(c[yte == 1].mean(0), color="#1f77b4", label="test successes")
        ax.plot(c[yte == 0].mean(0), color="#d62728", linestyle="--", label="test failures")
        ax.set_title(head)
        ax.set_xlabel("frame")
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("average score")
    axes[0].legend(fontsize=8)
    fig.suptitle(name.replace("_", " ") + " (seed 0)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "curves_%s.png" % name), dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--frames", type=int, default=16, help="use r3m_cache_<N>.pt (made by cache_r3m.py --frames N)")
    parser.add_argument("--cross", action="store_true", help="train on one task and test on the other")
    parser.add_argument("--cache", default=None, help="use this cache file instead")
    parser.add_argument("--out", default=None, help="write results here instead")
    parser.add_argument("--keep-unsure", action="store_true", help="keep cup failures marked 'not sure'")
    parser.add_argument("--rule", choices=RULES, default="drop",
                        help="flagging rule: drop (original), behind (new), or either")
    args = parser.parse_args()
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    cache_path = args.cache or os.path.join(
        HERE, "r3m_cache.pt" if args.frames == 16 else "r3m_cache_%d.pt" % args.frames)
    if not os.path.exists(cache_path):
        raise SystemExit("Missing %s. Make it first: py -3.12 week3/cache_r3m.py --frames %d" % (cache_path, args.frames))
    cache = torch.load(cache_path, weights_only=False)
    frames = cache["embedding"].shape[1]
    if args.out:
        out_dir = args.out
    else:
        folder = ("results" + ("" if args.rule == "drop" else "_" + args.rule) + ("_cross" if args.cross else "")
                  + ("" if frames == 16 else "_frames%d" % frames))
        out_dir = os.path.join(HERE, folder)
    os.makedirs(out_dir, exist_ok=True)
    drops, checked = label_check_drops(args.keep_unsure)
    robometer = load_robometer(frames)

    lines = ["Results from week3/run_heads.py (%d frames per attempt, flagging rule '%s'%s)" % (
        frames, args.rule, ", trained on one task and tested on the other" if args.cross else "")]
    lines.append("Cup failures checked on the label-check page: %d. Left out: %d." % (checked, len(drops)))
    lines.append("Robometer scores: %s." % ("%d attempts" % len(robometer) if robometer else "not run yet"))
    flag_rows = []
    pairs = [(TASKS[0], TASKS[1]), (TASKS[1], TASKS[0])] if args.cross else [(t, t) for t in TASKS]
    for train_task, test_task in pairs:
        evaluate(train_task, test_task, cache, drops, robometer, args.seeds, args.epochs, out_dir, flag_rows, lines,
                 args.rule)

    with open(os.path.join(out_dir, "flags.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # how: "rule" or "fallback". reference: the average training-success score at each frame
        # (used by the behind and either rules).
        writer.writerow(["task", "trained_on", "frames", "head", "seed", "episode_id", "label", "d", "s",
                         "flagged_frame", "scores", "rule", "how", "reference"])
        writer.writerows(flag_rows)
    text = "\n".join(lines)
    with open(os.path.join(out_dir, "results.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
