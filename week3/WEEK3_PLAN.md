# Week 3 plan

Claas said to start running experiments and send him a status update, so this week is about getting first numbers, not polishing the plan. The write-up comes at the end of the week, once there are results to report.

## What's left (updated Oct 21)

All the code is written. The R3M cache, the heads, and the first results are done (`week3/results/`). What's left is looking at pictures and running Robometer on TACC. All commands run from the repo folder in PowerShell.

1. **Push the branch** so TACC can clone it: `git push -u origin week3-r3m-cache`
2. **Check the 126 cup labels** (about 20 minutes, can be split between 3 people):
   `start week3\label_check\cups_label_check.html`, click an answer for each, press "Download results (CSV)", and move the file from Downloads into `week3\label_check\`.
3. **Mark where 40 failures start** (about 20 minutes each for 2 people):
   `start week3\onset\mark_onsets.html`, pick Annotator 1 or 2, mark your sheets, press "Download my answers", and move the file into `week3\onset\`. Don't look at each other's answers on the "both" sheets.
4. **Run Robometer on TACC** (Steps 2 and 3 below). Copy `week3/robometer_scores.jsonl` back to the laptop when it's done.
5. **Rerun the results** with the label check and Robometer included (about 10 minutes): `py -3.12 week3/run_heads.py`
6. **Check the failure starts against the heads:** `py -3.12 week3/check_onsets.py`

Steps 2, 3 and 4 don't depend on each other, so different people can do them at the same time.

## Two things we found while planning (read these first)

**1. The R3M cache runs on the laptop. It does not need TACC.**
We timed `week3/cache_r3m.py` on 8 episodes on this laptop's CPU. Reading the video took about 1 second per episode, and R3M took about 2.5 to 4 seconds. All 400 episodes take about **20 to 35 minutes**, not 6 to 8 TACC hours. After that, the heads train on 400 x 16 x 2,048 saved numbers, which takes seconds on a CPU. So:

- **The only job that needs TACC is Robometer** (a 4-billion-parameter model that needs a GPU).
- Retraining a head costs almost nothing, so we can try many versions. This is the answer to Claas's comment that 10–16 GPU-hours per run is too heavy: a head run is not 10–16 GPU-hours.

**2. The week-1 cup embeddings used the wrong picture size.**
R3M only resizes a picture when it is told the picture is not 224 x 224. Oven pictures are already 224 x 224, but cup pictures are 640 x 480, and week 1 passed them to R3M at full size. `cache_r3m.py` now resizes every picture to 224 x 224 and keeps the whole picture. Don't use `week1/r3m_two_tasks.pt` for anything.

## Checkpoints for the week

| # | Checkpoint | How you know it's done |
| --- | --- | --- |
| 1 | Full R3M cache | `week3/r3m_cache.pt` exists, shape (400, 16, 2048), 0 skipped |
| 2 | Logged in to TACC and found our allocation | We know the system name and allocation name |
| 3 | Robometer test job ran on 8 episodes | We have a measured number of seconds per episode |
| 4 | First head results on both tasks | Score-curve plots plus one results table (Step 4) |
| 5 | 40 onset labels | `onset_labels.csv` filled in, agreement on the shared 10 checked |
| 6 | Robometer on all 400 | `week3/robometer_scores.jsonl` has 400 lines |
| 7 | Status write-up sent to Claas | End of the week |

Checkpoints 1, 4 and 5 don't depend on TACC. If TACC is slow or broken, keep going on those.

**One way to split the work** (change it however you like):
- Hiya: TACC and Robometer (Steps 2–3), since your account works.
- Person 2: R3M cache and the first heads (Steps 1 and 4).
- Person 3: onset sheets and labels (Step 5). Hiya labels the 10 shared sheets as the second annotator.

---

## Step 0: Put these files in the repo (10 minutes)

1. Copy the `week3` folder into `C:\Users\sunee\fri-oopsiedata-week1`, so it sits next to `week2`.
2. Install scikit-learn, which the laptop doesn't have yet:
   ```powershell
   py -3.12 -m pip install scikit-learn matplotlib
   ```
3. Keep the big cache file out of git. Add this line to `.gitignore`:
   ```
   week3/*.pt
   ```
4. Commit and push, so TACC can `git clone` the same code.

## Step 1: R3M cache on the laptop (about 30 minutes, mostly waiting)

From the repo folder:

```powershell
py -3.12 week3/cache_r3m.py --limit 2
```

That's the 8-episode timing test. If it finishes, run all 400:

```powershell
py -3.12 week3/cache_r3m.py
```

It prints progress every 20 episodes and ends with `Saved (400, 16, 2048)`. If a video can't be read, the script skips it and lists it, so one bad file can't throw away the rest.

**Sanity check (5 minutes).** In week 2 we checked raw pixels: matched success/failure pairs were close early and further apart late. Check that the same holds in R3M numbers for oven pair 0:

```powershell
py -3.12 -c "import torch; c=torch.load('week3/r3m_cache.pt', weights_only=False); e=c['embedding']; m=c['episodes']; i=[k for k,x in enumerate(m) if x['task']=='open_oven' and x['pair_id']=='0' and x['session_id'].endswith('18-14-02')]; print([m[k]['label'] for k in i]); d=1-torch.nn.functional.cosine_similarity(e[i[0]], e[i[1]], dim=1); print([round(float(x),3) for x in d])"
```

It prints 16 distances, one per picture. Small numbers at the start and bigger ones at the end mean the embeddings separate the way the proposal assumes. Write down what you see either way. A flat line is a result too.

## Step 2: TACC basics (about 1 hour the first time)

**How TACC works, in short.** You SSH into a *login node*. Login nodes are only for editing files, copying data, and submitting jobs. Never run Python models there. Real work runs on *compute nodes*, which you get in one of two ways:
- `idev` gives you an interactive shell on a compute node. Use it for installing and debugging.
- `sbatch file.slurm` puts a script in the queue, and it runs when a GPU is free. Use it for real runs. Output goes to a `.out` file.

Every GPU hour is charged to your allocation in "SUs" (service units).

**Storage.** You have three folders, and putting things in the wrong one is the most common beginner mistake:

| Folder | Size | Use it for |
| --- | --- | --- |
| `$HOME` | 10 GB | Small config only. Don't install Python packages or models here |
| `$WORK` | 1 TB, never purged | Code, the Python environment, our data, results |
| `$SCRATCH` | Very large, **files unused for 10 days get deleted** | Download caches and model weights (they can be downloaded again) |

**2a. Log in** from PowerShell on the laptop:

```powershell
ssh YOUR_TACC_USERNAME@ls6.tacc.utexas.edu
```

It asks for your TACC password and then the 6-digit code from the TACC Token app.

**2b. Find your allocation.** Run this on the login node:

```bash
/usr/local/etc/taccinfo
```

It lists your project or allocation name(s) and SU balance. Copy the project name into the `#SBATCH -A YOUR_ALLOCATION` line of both `.slurm` files.

> These steps assume **Lonestar6** (`ls6`). If `taccinfo` shows your allocation is on a different system (for example Vista), stop and check with your peer mentor first. Vista uses ARM processors, and some of Robometer's packages (xformers, decord) are hard to install there.

**2c. Copy the code and data to TACC.**

On TACC, clone the repo into `$WORK`:

```bash
cd $WORK
git clone https://github.com/hiyasvyas/failure-detector-for-oopsiedata.git
```

If the repo is private, git asks for a username and password. The password must be a GitHub personal access token, not your GitHub password.

On the laptop (PowerShell), pack the two tasks into one file and send it. That's about 370 MB, and packing first means one upload and one Token code:

```powershell
tar -czf $HOME\oopsie_two_tasks.tgz -C C:\Users\sunee\oopsiedata-v0.1 labs/Araya/flower_open_oven labs/Harvard_EML/stack-and-pack-cups
scp $HOME\oopsie_two_tasks.tgz YOUR_TACC_USERNAME@ls6.tacc.utexas.edu:
```

Back on TACC:

```bash
mkdir -p $WORK/oopsie
tar -xzf ~/oopsie_two_tasks.tgz -C $WORK/oopsie
rm ~/oopsie_two_tasks.tgz
ls $WORK/oopsie/labs
```

The data is private. Keep it in your own `$WORK` folder and don't put it anywhere shared.

## Step 3: Robometer on TACC

**3a. Install it once** on a GPU dev node. The dev queue allows up to 2 hours per session and one session at a time:

```bash
idev -p gpu-a100-dev -N 1 -n 1 -t 1:00:00
# wait for a prompt that starts with a node name like c3xx-xxx, then:
bash $WORK/failure-detector-for-oopsiedata/week3/tacc/setup_robometer.sh
exit
```

The script installs `uv`, clones Robometer into `$WORK/robometer`, installs its packages into `$WORK/robometer/.venv`, checks the GPU is visible, and downloads the Robometer-4B weights to `$SCRATCH`. Expect 15–30 minutes. The last lines should read `cuda True` and `Setup finished.`

**3b. Test job on 8 episodes** (from the repo folder on TACC):

```bash
cd $WORK/failure-detector-for-oopsiedata
sbatch week3/tacc/robometer_test.slurm
squeue -u $USER          # PD = waiting in the queue, R = running
cat robometer-test.*.out # read the output once it's done
```

Write down three numbers from the `.out` file: how long the model took to load, the seconds per episode, and how many progress scores came back per episode. It should be 16. If it isn't, tell Claas, because the shared alarm expects one score per picture.

**3c. Full run.** Set `#SBATCH -t` in `robometer_all.slurm` to about twice what the test predicts, then:

```bash
sbatch week3/tacc/robometer_all.slurm
```

If it runs out of time, run the same `sbatch` command again. Episodes that already have scores are skipped. Copy the results back to the laptop from PowerShell:

```powershell
scp YOUR_TACC_USERNAME@ls6.tacc.utexas.edu:WORK_PATH/failure-detector-for-oopsiedata/week3/robometer_scores.jsonl week3/
```

Replace `WORK_PATH` with what `echo $WORK` prints on TACC.

**Useful commands:** `squeue -u $USER` (my jobs), `scancel JOBID` (stop a job), `/usr/local/etc/taccinfo` (SUs left).

**Cost:** `gpu-a100-small` is one A100 at 1.5 SUs per hour, and `gpu-a100-dev` is 3 SUs per hour. Install, test and the full run together should be a few SUs. The test job tells you the real number, which is exactly what Claas asked us to measure.

## Step 4: First head results on the laptop (the main experiment work)

Train everything **per task** for now (oven alone, cups alone). Each step below runs in seconds once the cache exists.

**4a. Make a calibration set without touching the test set.** The alarm threshold has to be picked on successes the head never trained on. There are only 34 oven and 56 cup training successes, so don't carve off a small separate set. Instead:
- Split the training episodes into 5 groups: by `pair_id` for the oven (all training oven episodes come from one session) and by `session_id` for the cups.
- Train the head 5 times, each time leaving one group out, and score the left-out group.
- Now every training success has a score from a head that never saw it. Pick the alarm setting on those scores.
- Train once more on all training data, and use that head on the test set.

**4b. Episode-level baselines.** These give one success/failure answer per attempt and no picture number: a majority-class guess, logistic regression on the mean of the 16 embeddings, and a small classifier on all 16 embeddings. Report episode F1 on the test split.

**4c. Head D (no training).** Store every picture of every training success. Score a picture by its distance to the nearest stored picture *from a different episode*, then map that distance to 0–1 using training successes only. A typical success picture should land near 1.

**4d. Head A, two versions.** This answers our question to Claas about whether Head A should see failures. Train both, since it's cheap:
- **A1, successes only** (the original control): target `t/15` for picture `t` of each success, and no target on failures.
- **A2, also uses failures** (Claas's suggestion): the same targets on successes, plus a target of 0 on the **last** picture of each failure, because a failed attempt did not finish. Other failure pictures get no target. A2 uses only the label we already have, and needs no onset times.

Train each 3 times with different seeds.

**4e. The alarm, one rule for every head.** It fires at the first picture where the score has fallen at least `d` below the best score so far in the attempt and has stayed down for `s` pictures in a row. If it never fires and the final score is below 0.5, it fires at picture 15. Try `d` in {0.05, 0.10, 0.20} and `s` in {2, 3, 4}, and keep the setting whose false-alarm rate on the calibration successes (4a) is closest to 10%. **Freeze it before looking at test failures.**

**4f. What to save for the status update:**
- One plot per task: the mean score over the 16 pictures for test successes and for test failures, one line per head. This is likely the most useful figure.
- One table per task with a row per head: false alarms on test successes, the share of test failures with an alarm, the median alarm picture, and how many alarms came before picture 13.
- The comparison between heads doesn't need onset times, because the alarm picture is measured on the same episodes for every head. Onset times (Step 5) are what tell us whether an "early" alarm came *after* the attempt actually went wrong.

**Decision point from the proposal:** if A1 already fires before the last 3 pictures on most test failures, the preference claim is in doubt. That's a real result: write it down and tell Claas.

**Stretch, if there's time:** Head B. Use A1's targets plus a ranking loss: for each success/failure pair, the mean of the success's 16 scores should be at least 0.2 above the failure's mean. For the oven, each `pair_id` within a session holds two runs from the same start under two policy settings. Use the pairs that contain one success and one failure. For the cups, pair a success and a failure from the same session. It also trains in seconds.

**Things to watch:**
- Oven: 91 of 148 failures ran to the 400-step limit, against 2 of 52 successes. Using 16 pictures hides the length, but a failed attempt's 16 pictures are spread over 33 seconds, while a success's are spread over about 9. Check that a head isn't simply learning "the arm moved slowly".
- Cups: the last picture of a failure often looks normal because the arms go home, so the "final score below 0.5" rule may miss cup failures.
- Test sets are small: 18 test successes per task. A 10% false-alarm rate means about 2 attempts, so report counts, not only percentages.

## Step 5: Onset labels (about 2 hours of looking, split between two people)

```powershell
py -3.12 week3/make_onset_sheets.py
```

This picks 20 failed **test** attempts per task (40 total, fixed seed) and saves one picture sheet each to `week3/onset/sheets/`. Each sheet shows the same 16 pictures the heads see, numbered 0–15. It also writes `week3/onset/mark_onsets.html`, the page you mark them on, and `week3/onset/onset_sheets.csv`, which says who marks which sheet. (Already done; you only need to rerun it if the sheets get deleted.)

- Open the page, pick Annotator 1 or 2, and for each sheet type the number of the **first frame where you can see the attempt has gone wrong**: a cup tips over, the gripper misses the handle and moves away, or the door stops moving. Many oven failures just run out of time, so if there's no clear moment, tick "No clear moment" and say why in the note. That's a useful answer too.
- 10 sheets are marked "both". Each person marks those **without looking at the other's answer**.
- Press "Download my answers" and move `onset_annotator1.csv` or `onset_annotator2.csv` into `week3/onset/`. Then `py -3.12 week3/check_onsets.py` checks agreement and compares the marks with each head's flagged frame.
- Agreement rule, from the proposal: if the two people disagree by more than 2 pictures on more than 3 of the 10 shared sheets, don't use the hand labels. Report alarm pictures only.
- These sheets are also our check on the cup labels. If a sheet doesn't look like a failure, say so in `notes`.

## Our three questions, answered as far as we can

- **When do we move work off the laptop?** Only Robometer goes to TACC. The R3M cache (about 30 minutes) and all the heads run on the laptop.
- **Is TACC the right place for Robometer?** Yes. It needs an NVIDIA GPU, and one A100 (40 GB) on `gpu-a100-small` is plenty for a 4-billion-parameter model doing inference only.
- **Can we train on cup failures marked "pending human review"?** The full note reads "Imported from the source rollout marker; taxonomy pending human review." The *pending* part is the taxonomy: what kind of failure it was and how bad. That field is empty for the oven failures too, and we don't use it. The success/failure mark itself came from the Harvard lab's own rollout marker, and it matches the `-fail-` in all 126 filenames. So yes, train on them. Our 7 watched cup failures and the 20 onset sheets are our spot check, and it's still worth confirming with Claas in the status update.

## If something goes wrong

| If this happens | Do this |
| --- | --- |
| `uv sync` fails on tensorflow or tensorflow-datasets | Robometer's own `pyproject.toml` says those are only for uploading datasets. Rerun with `uv sync --extra robometer --no-install-package tensorflow --no-install-package tensorflow-datasets --no-install-package tf-keras` |
| Robometer still won't install after two tries | Ask the peer mentor or open a TACC help ticket. Meanwhile Steps 1, 4 and 5 don't need it |
| Robometer isn't running by next week | Drop it and say so. It's a comparison point, and the Head A vs. Head B test doesn't need it |
| Robometer is slow | Score only the test split (148 episodes). Results are saved after every episode |
| The queue is long | Use `gpu-a100-small` (1 GPU). Nothing else needs TACC |
| A1 already alarms early | The preference claim is in doubt. Report it, then compare A2 and B anyway |
| Onset labels disagree | Follow the agreement rule above and report alarm pictures only |

## Afterwards: the status write-up

Write it **yourselves**, in your own words. Claas told us AI-sounding writing works against us. Short and plain is fine. While you work this week, collect what you'll need to answer his feedback:

- **Alarm:** define it in one sentence the first time you use it (Step 4e).
- **Robometer's numbers:** for the 0.98 → 0.48 and 0.92 → −0.14 figures, say which paper and table they come from, and what VOC and Kendall's τ each measure, in a sentence.
- **"Trajectories share the scene until the failure":** replace it with what we measured in week 2, on the 0–255 pixel scale. Oven pairs differ by 4.7–7.4 on the first 3 pictures and 6.4–14.2 on the last 3. Cup pairs differ by 5.4–11.9 early and 14.1–25.1 late. Add the R3M check from Step 1.
- **Why several heads:** each head changes one training signal and keeps everything else fixed, so a difference in alarm timing comes from that one signal.
- **Why 16 pictures:** every attempt gets the same number of pictures, so video length can't stand in for failure.
- **Head A and failures:** A1 vs. A2 (Step 4d).
- **Head C's "episode mean":** the mean of the 16 per-picture scores for one attempt.
- **Compute:** measured R3M time, measured Robometer time, and the fact that head training takes seconds.
- **Work plan:** the checkpoints and the fallback table above.
