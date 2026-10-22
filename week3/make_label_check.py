"""Make one web page for checking the cup failure labels by eye.

All 126 cup failures say "taxonomy pending human review". The success/failure mark
came from the lab's rollout marker, and nobody has looked at them all. This page shows
the same 16 pictures the heads see for every cup failure, with the middle and end
pictures included, so checking one takes a few seconds instead of watching a video.
A few successes sit at the top so you can see what a finished attempt looks like.

Run:   py -3.12 week3/make_label_check.py
Open:  week3/label_check/cups_label_check.html in a browser
Click an answer for each attempt, type your name, then "Download results" saves
cups_label_check_<name>.csv. To share the page, zip the whole label_check folder
(the page needs its pictures/ folder next to it).
"""

import argparse
import csv
import html
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

PAGE_TOP = """<!doctype html>
<html><head><meta charset="utf-8"><title>Cup label check</title>
<style>
body { font-family: Arial, sans-serif; max-width: 1000px; margin: 24px auto; padding: 0 16px; color: #1a1a1a; }
.item { border-top: 1px solid #ccc; padding: 12px 0; }
.item img { width: 100%%; max-width: 720px; display: block; }
.done { background: #f3f8ee; }
.bar { position: sticky; top: 0; background: #fff; padding: 8px 0; border-bottom: 1px solid #ccc; }
label { margin-right: 16px; }
</style></head><body>
<h1>Cup label check</h1>
<p>Each sheet shows the 16 pictures our heads see, numbered 0 to 15, from start to end.
For each failure, pick whether it really looks failed. Look at the middle and the end:
a success ends with the cups stacked and packed. Your answers stay in this browser
until you press "Download results".</p>
<p>If three of us split the work: the first person checks Failures 1 to 42, the second 43 to 84,
and the third 85 to 126. Type your name in the box at the top before you download, and send
the CSV to Hiya.</p>
<div class="bar">Your name: <input id="name" type="text" size="12">
&nbsp; <span id="count">0</span> of %(n)d checked
<button onclick="download()">Download results (CSV)</button>
<span id="name-error" style="color:#b00020;font-size:13px"></span></div>
<h2>What a success looks like (%(n_ref)d examples)</h2>
"""

PAGE_BOTTOM = """<script>
var KEY = "cup-label-check";
var saved = {};
try { saved = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) {}
function store() { try { localStorage.setItem(KEY, JSON.stringify(saved)); } catch (e) {} }
function refresh() {
  var n = 0;
  document.querySelectorAll(".item.failure").forEach(function (item) {
    var answer = (saved[item.dataset.id] || {}).answer;
    item.classList.toggle("done", !!answer);
    if (answer) n++;
  });
  document.getElementById("count").textContent = n;
}
document.querySelectorAll(".item.failure").forEach(function (item) {
  var id = item.dataset.id, entry = saved[id] || {};
  item.querySelectorAll("input[type=radio]").forEach(function (radio) {
    radio.checked = radio.value === entry.answer;
    radio.addEventListener("change", function () {
      saved[id] = saved[id] || {}; saved[id].answer = radio.value; store(); refresh();
    });
  });
  var note = item.querySelector("input[type=text]");
  note.value = entry.note || "";
  note.addEventListener("input", function () {
    saved[id] = saved[id] || {}; saved[id].note = note.value; store();
  });
});
var nameBox = document.getElementById("name");
try { nameBox.value = localStorage.getItem(KEY + "-name") || ""; } catch (e) {}
nameBox.addEventListener("input", function () {
  document.getElementById("name-error").textContent = "";
  try { localStorage.setItem(KEY + "-name", nameBox.value); } catch (e) {}
});
function download() {
  // Each person's file gets their name, so several files can sit in week3/label_check/ together.
  var name = nameBox.value.trim().replace(/[^A-Za-z0-9_-]/g, "");
  if (!name) { document.getElementById("name-error").textContent = "Type your name first."; return; }
  var lines = ["episode_id,session_id,answer,note,checked_by"];
  document.querySelectorAll(".item.failure").forEach(function (item) {
    var entry = saved[item.dataset.id] || {};
    var note = (entry.note || "").replace(/"/g, '""');
    lines.push([item.dataset.id, item.dataset.session, entry.answer || "", '"' + note + '"', name].join(","));
  });
  var link = document.createElement("a");
  link.href = URL.createObjectURL(new Blob([lines.join("\\n") + "\\n"], {type: "text/csv"}));
  link.download = "cups_label_check_" + name + ".csv";
  link.click();
}
refresh();
</script></body></html>
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=os.path.dirname(HERE), help="repo folder that holds week2/")
    parser.add_argument("--references", type=int, default=3, help="successes shown at the top")
    args = parser.parse_args()

    sys.path.insert(0, os.path.join(args.repo, "week2"))
    sys.path.insert(0, HERE)
    from loader import load_episode
    from make_onset_sheets import sheet_image

    with open(os.path.join(args.repo, "week2", "episode_index.csv"), encoding="utf-8") as f:
        rows = [row for row in csv.DictReader(f) if row["task"] == "stack_cups"]
    failures = sorted((row for row in rows if row["label"] == "failure"), key=lambda r: (r["session_id"], r["episode_id"]))
    successes = []
    for row in rows:  # one success from each of the first few sessions
        if row["label"] == "success" and row["session_id"] not in {s["session_id"] for s in successes}:
            successes.append(row)
    successes = successes[:args.references]

    out = os.path.join(HERE, "label_check")
    pictures = os.path.join(out, "pictures")
    os.makedirs(pictures, exist_ok=True)

    parts = [PAGE_TOP % {"n": len(failures), "n_ref": len(successes)}]
    for kind, group in (("success", successes), ("failure", failures)):
        if kind == "failure":
            parts.append("<h2>The %d cup failures</h2>" % len(failures))
        for number, row in enumerate(group, start=1):
            name = "%s_%03d.jpg" % (kind, number)
            sheet_image(load_episode(row["h5_path"], "stack_cups")["frames"]).save(
                os.path.join(pictures, name), quality=80)
            ident = html.escape(row["episode_id"])
            session = html.escape(row["session_id"])
            parts.append('<div class="item %s" data-id="%s" data-session="%s">' % (kind, ident, session))
            parts.append("<h3>%s %d of %d</h3><p>%s<br>session %s</p>" % (
                "Success" if kind == "success" else "Failure", number, len(group), ident, session))
            parts.append('<img src="pictures/%s" loading="lazy" alt="16 pictures from %s">' % (name, ident))
            if kind == "failure":
                radio = '<label><input type="radio" name="%s" value="%s"> %s</label>'
                parts.append("<p>" + radio % (ident, "failed", "Yes, it failed")
                             + radio % (ident, "looks_success", "No, it looks like a success")
                             + radio % (ident, "unsure", "Not sure")
                             + '<input type="text" placeholder="note (what went wrong, which picture)" size="40"></p>')
            parts.append("</div>")
            print(kind, number, row["episode_id"])
    parts.append(PAGE_BOTTOM)

    page = os.path.join(out, "cups_label_check.html")
    with open(page, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    print("Wrote", page)


if __name__ == "__main__":
    main()
