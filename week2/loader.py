"""Load OopsieData episodes for the two week-2 tasks.

Reads the success mark already stored in each episode file and returns 16
evenly spaced pictures from the main camera. Does not modify episode files
or videos.
"""

import json
import os

import h5py
import imageio.v3 as iio
import numpy as np

DATA = r"C:\Users\sunee\oopsiedata-v0.1"

TASKS = {
    "open_oven": os.path.join(DATA, r"labs\Araya\flower_open_oven"),
    "stack_cups": os.path.join(DATA, r"labs\Harvard_EML\stack-and-pack-cups"),
}


def sample_16(video_path):
    """16 picture numbers spread from the first frame to the last, including both ends."""
    n = 0
    for _ in iio.imiter(video_path):
        n += 1
    if n < 2:
        raise RuntimeError("Video has too few frames: " + video_path)
    wanted = [int(x) for x in np.linspace(0, n - 1, 16)]
    wanted_set = set(wanted)
    grabbed = {}
    for index, frame in enumerate(iio.imiter(video_path)):
        if index in wanted_set and index not in grabbed:
            grabbed[index] = frame
        if len(grabbed) == len(wanted_set):
            break
    missing = [i for i in wanted if i not in grabbed]
    if missing:
        raise RuntimeError("Could not read frames %s from %s" % (missing, video_path))
    frames = np.stack([grabbed[i] for i in wanted])
    return frames, wanted, n


def _main_video(h5_path, h5_file):
    folder = os.path.dirname(h5_path)
    paths = h5_file["observations/video_paths"]
    if "image" in paths:
        camera = "image"
    elif "base_image" in paths:
        camera = "base_image"
    else:
        raise RuntimeError("No main camera in " + h5_path)
    name = paths[camera][()]
    if isinstance(name, bytes):
        name = name.decode()
    return camera, os.path.join(folder, str(name))


def _notes_and_session(task, h5_path, notes_text):
    text = "" if notes_text is None else str(notes_text)
    if task == "open_oven":
        notes = json.loads(text) if text.strip().startswith("{") else {}
        session_id = str(notes.get("session_stamp", "unknown"))
        pair_id = notes.get("pair_id")
        return session_id, pair_id, text
    session_id = os.path.basename(os.path.dirname(h5_path))
    return session_id, None, text


def read_record(h5_path, task):
    """Labels and paths only. Does not decode the video."""
    with h5py.File(h5_path, "r") as h:
        groups = list(h["episode_annotations"].keys())
        if len(groups) != 1:
            raise RuntimeError("Expected one annotation group in %s, found %s" % (h5_path, groups))
        operator = groups[0]
        attrs = h["episode_annotations"][operator].attrs
        success = float(attrs["success"])
        camera, video = _main_video(h5_path, h)
        session_id, pair_id, notes = _notes_and_session(task, h5_path, attrs.get("additional_notes", ""))
        return {
            "task": task,
            "episode_id": str(h.attrs.get("episode_id", os.path.splitext(os.path.basename(h5_path))[0])),
            "h5_path": h5_path,
            "video_path": video,
            "camera": camera,
            "instruction": str(h.attrs.get("language_instruction", "")),
            "operator": operator,
            "success": success,
            "label": "success" if success == 1.0 else "failure" if success == 0.0 else "other",
            "session_id": session_id,
            "pair_id": pair_id,
            "notes": notes,
            "video_exists": os.path.isfile(video),
        }


def index_task(task):
    folder = TASKS[task]
    records = []
    for dirpath, _, files in os.walk(folder):
        for name in files:
            if name.endswith(".h5"):
                records.append(read_record(os.path.join(dirpath, name), task))
    records.sort(key=lambda row: row["episode_id"])
    return records


def index_all():
    rows = []
    for task in TASKS:
        rows.extend(index_task(task))
    return rows


def load_episode(h5_path, task):
    """Record plus 16 pictures. frames shape is (16, height, width, 3)."""
    record = read_record(h5_path, task)
    if not record["video_exists"]:
        raise RuntimeError("Missing video: " + record["video_path"])
    frames, indexes, n_frames = sample_16(record["video_path"])
    record["n_frames"] = n_frames
    record["frame_indexes"] = indexes
    record["frames"] = frames
    return record


def session_split(records, test_fraction=0.2, seed=0):
    """Assign every episode in a session to the same side.

    Oven has two recording sessions, so one session is the test set.
    Cups have many log folders; about 20 percent of those folders are held out.
    """
    by_task = {}
    for row in records:
        by_task.setdefault(row["task"], {}).setdefault(row["session_id"], []).append(row)

    assignment = {}
    rng = np.random.default_rng(seed)
    for task, sessions in by_task.items():
        names = sorted(sessions)
        if task == "open_oven":
            test_names = {names[-1]}
        else:
            order = list(names)
            rng.shuffle(order)
            n_test = max(1, int(round(len(order) * test_fraction)))
            test_names = set(order[:n_test])
        for name in names:
            side = "test" if name in test_names else "train"
            for row in sessions[name]:
                assignment[row["h5_path"]] = side
    return assignment
