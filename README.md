# Spotting the moment a robot fails on OopsieData

Annie Li, Phoebe Wang, and Hiya Vyas. FRI II Robot Learning, The University of Texas at Austin.

This repo follows `proposal/project_proposal.tex`. The goal is to name the picture where a recorded robot attempt starts to fail. A frozen R3M encoder turns each picture into 2,048 numbers. Four small heads will later share one alarm. No head has been trained. The Hugging Face dataset was not edited, and nothing was uploaded back to it.

## What is done

We took a look at OopsieData v0.1.
The success mark is inside each episode file at `episode_annotations/<operator name>/success`. `1.0` means the attempt worked. `0.0` means it failed. The labs wrote those marks. The file does not say which picture the failure started on.

We chose 2 tasks that are different skills, on different robots, and both already have successes and failures:

| Task | Episodes | Successes | Failures | What we use |
| --- | --- | --- | --- | --- |
| Araya, "Open the oven" | 200 | 52, operator `LN` | 148, operator `LN` | The table-view video, `image` |
| Harvard EML, "Stack and pack the cups." | 200 | 74, operator `HXS` | 126, operator `imported_source_label` | The base-view video, `base_image` |

The oven folder has 600 videos because each attempt was saved from three cameras. One attempt is one `.h5` file plus its main-camera `.mp4`. 

R3M is installed from https://github.com/facebookresearch/r3m. The weights are `resnet50`, stored locally in `.r3m` and not in this repo. R3M was not trained.

Two R3M runs are in this repo:

- `week1/r3m_one_frame.pt` is the first run. One picture from a 13-frame oven clip. That clip is too short to be an experiment task. The file holds one vector of 2,048 numbers.
- `week1/r3m_two_tasks.pt` is the second run, on the two real tasks. Four attempts: one oven success, one oven failure, one cup success, one cup failure. The script counted the pictures in each video, kept 16 pictures spread from the first to the last, and saved one vector per picture. Each of the four blocks has shape 16 by 2048. The other 396 attempts have not been through R3M.

The script that made the second file is `week1/run_r3m_two_tasks.py`. To look at the saved numbers:

```powershell
py -3.12 -c "import torch; x=torch.load(r'week1/r3m_two_tasks.pt', weights_only=False); print(list(x)); v=x['open_oven_failure']['embedding'][0]; print(v.shape); print(round(float(v.norm()), 2))"
```

Robometer's code was cloned from https://github.com/robometer/robometer. It's not in this repo, we need TACC for this.

## Files in this repo

| File | What it is |
| --- | --- |
| `proposal/project_proposal.tex` | The FRI proposal |
| `week1/week1_label_note.txt` | The first two episode files we opened |
| `week1/task_counts.txt` | Episode counts by folder. "fail 479" there is a path-name count, not a count of failed attempts |
| `week1/r3m_result.txt` | The one-frame R3M run, in words |
| `week1/r3m_one_frame.pt` | One 2,048-number vector from the short clip |
| `week1/run_r3m_two_tasks.py` | How the four real attempts were read and embedded |
| `week1/r3m_two_tasks.pt` | Four attempts, 16 vectors each |
| `download_data.py` | Downloads the WebVisualizer dataset into `data/` after `hf auth login`. That folder is gitignored |
| `week2/loader.py` | Reads the success mark and returns 16 frames from the main camera |
| `week2/run_week2.py` | Ran that loader on 30 attempts and saved the pictures |
| `week2/episode_index.csv` | All 400 episodes, with the label, session, and train or test split |
| `week2/week2_report.txt` | The counts and the early-versus-late check |
| `week2/watch.html` | The 30 attempts, start, middle, and end. Open the pictures in `week2/frames` |

## Setup for download_data.py

```
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -U huggingface_hub datasets
hf auth login
python download_data.py
```

## Week 2

The loader is `week2/loader.py`. It reads the success mark already in the episode file and keeps 16 pictures from the main camera, spread from the first picture to the last. We ran it on 15 oven attempts and 15 cup attempts. A success and a failure of the oven start on the same table. The success ends with the door open. The failure ends with the door still shut. Cup failures say "pending human review" in the file, and the mark matches the "-fail-" in the filename. The mistake shows up in the middle of the cup video. The last frame often looks normal again because the arms go home.

## What is next

The full R3M cache, Robometer inference, and the four heads belong on TACC. We still need to learn how to use it and TACC and when to move this work off the laptop. Failure onset times are not in the files we opened. If we need timing, two people need to later mark about 40 failed videos, with 10 marked by both.
