"""Show where the cup success/failure marks come from.

Prints what is stored inside one cup failure file and one cup success file, then
checks every cup episode: how many failures have the "pending human review" note,
whether the failure mark matches "-fail-" in the file name, and whether the
failure type is filled in. The oven is checked too, for comparison.
Only reads the episode files. Does not change them.

Run:  py -3.12 week3/show_cup_labels.py
"""

import csv
import json
import os

import h5py

HERE = os.path.dirname(os.path.abspath(__file__))
INDEX = os.path.join(os.path.dirname(HERE), "week2", "episode_index.csv")


def annotation(h5_path):
    with h5py.File(h5_path, "r") as h:
        group = list(h["episode_annotations"].keys())[0]
        attrs = h["episode_annotations"][group].attrs
        fields = {key: str(attrs[key]) for key in attrs.keys()}
    fields["annotation_group"] = group
    return fields


def show(row):
    print("File:", row["h5_path"])
    print("  inside the file, at episode_annotations/%s:" % annotation(row["h5_path"])["annotation_group"])
    for key, value in annotation(row["h5_path"]).items():
        if key != "annotation_group":
            print("    %-16s = %s" % (key, value[:150]))
    print()


def main():
    with open(INDEX, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    cups = [row for row in rows if row["task"] == "stack_cups"]

    print("=== One cup failure")
    show(next(row for row in cups if row["label"] == "failure"))
    print("=== One cup success")
    show(next(row for row in cups if row["label"] == "success"))

    print("=== Checking every episode")
    for task in ("stack_cups", "open_oven"):
        failures = [row for row in rows if row["task"] == task and row["label"] == "failure"]
        successes = [row for row in rows if row["task"] == task and row["label"] == "success"]
        notes = [annotation(row["h5_path"]) for row in failures]
        pending = sum("pending human review" in n.get("additional_notes", "") for n in notes)
        no_type = sum(json.loads(n["taxonomy"]).get("failure_category") == [] for n in notes)
        print(task)
        print("  failures: %d, successes: %d" % (len(failures), len(successes)))
        print("  failures with the 'pending human review' note: %d" % pending)
        print("  failures with no failure type filled in: %d of %d" % (no_type, len(failures)))
        if task == "stack_cups":
            fail_named = [row for row in cups if "-fail-" in row["episode_id"]]
            print("  file names containing '-fail-': %d, and all of those are marked failure: %s" % (
                len(fail_named), all(row["label"] == "failure" for row in fail_named)))
            print("  successes with '-fail-' in the name: %d" % sum("-fail-" in row["episode_id"] for row in successes))


if __name__ == "__main__":
    main()
