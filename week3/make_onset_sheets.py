"""Make picture sheets and a web page for marking where each failure starts.

Picks 20 failed test attempts per task (40 total, fixed seed). Each sheet shows the
same 16 pictures the heads see, numbered 0 to 15. 10 of the 40 are marked "both":
both annotators mark those without looking at each other's answer, so we can check
agreement. The other 30 are split between annotator 1 and annotator 2.

Run:   py -3.12 week3/make_onset_sheets.py
Open:  week3/onset/mark_onsets.html in a browser, pick who you are, and mark your sheets.
"Download my answers" saves onset_annotator1.csv or onset_annotator2.csv. Put it in
week3/onset/, then run week3/check_onsets.py.
"""

import argparse
import csv
import html
import os
import random
import sys

from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
TILE_WIDTH = 240

PAGE_TOP = """<!doctype html>
<html><head><meta charset="utf-8"><title>Mark where the failure starts</title>
<style>
body { font-family: Arial, sans-serif; max-width: 1000px; margin: 24px auto; padding: 0 16px; color: #1a1a1a; }
.item { border-top: 1px solid #ccc; padding: 12px 0; }
.item img { width: 100%; max-width: 720px; display: block; }
.done { background: #f3f8ee; }
.bar { position: sticky; top: 0; background: #fff; padding: 8px 0; border-bottom: 1px solid #ccc; }
label { margin-right: 16px; }
</style></head><body>
<h1>Mark where the failure starts</h1>
<p>Each sheet is one failed attempt: the 16 frames our heads see, numbered 0 to 15. Write the number
of the <b>first frame where you can see the attempt has gone wrong</b>, for example a cup tips over,
the gripper misses the handle and moves away, or the door stops moving. If there is no clear moment
(for example the robot just never gets the door open), tick "No clear moment" and say why in the note.</p>
<p>On the sheets marked <b>both</b>, the other annotator marks the same attempt. Don't look at each
other's answers. Your answers stay in this browser until you press "Download my answers".</p>
<div class="bar">I am:
<select id="who" onchange="show()"><option value="">pick one</option>
<option value="annotator1">Annotator 1</option><option value="annotator2">Annotator 2</option></select>
&nbsp; <span id="count">0</span> of <span id="total">0</span> marked
<button onclick="download()">Download my answers (CSV)</button></div>
"""

PAGE_BOTTOM = """<script>
function key() { return "onset-" + document.getElementById("who").value; }
function load() { try { return JSON.parse(localStorage.getItem(key()) || "{}"); } catch (e) { return {}; } }
function save(data) { try { localStorage.setItem(key(), JSON.stringify(data)); } catch (e) {} }
function mine(item, who) { return who && (item.dataset.who === "both" || item.dataset.who === who); }
function refresh() {
  var who = document.getElementById("who").value, data = load(), n = 0, total = 0;
  document.querySelectorAll(".item").forEach(function (item) {
    if (!mine(item, who)) return;
    total++;
    var e = data[item.dataset.id] || {};
    var done = (e.onset !== undefined && e.onset !== "") || e.none;
    item.classList.toggle("done", !!done);
    if (done) n++;
  });
  document.getElementById("count").textContent = n;
  document.getElementById("total").textContent = total;
}
function show() {
  var who = document.getElementById("who").value, data = load();
  document.querySelectorAll(".item").forEach(function (item) {
    item.style.display = mine(item, who) ? "" : "none";
    var e = data[item.dataset.id] || {};
    item.querySelector(".onset").value = e.onset === undefined ? "" : e.onset;
    item.querySelector(".none").checked = !!e.none;
    item.querySelector(".note").value = e.note || "";
    item.querySelector(".error").textContent = "";
  });
  refresh();
}
document.querySelectorAll(".item").forEach(function (item) {
  function store() {
    if (!document.getElementById("who").value) return;
    var raw = item.querySelector(".onset").value.trim(), error = item.querySelector(".error");
    error.textContent = "";
    if (raw !== "" && !/^(1[0-5]|[0-9])$/.test(raw)) { error.textContent = "Use a frame number from 0 to 15."; return; }
    var data = load();
    data[item.dataset.id] = {onset: raw, none: item.querySelector(".none").checked, note: item.querySelector(".note").value};
    save(data); refresh();
  }
  item.querySelector(".onset").addEventListener("input", store);
  item.querySelector(".none").addEventListener("change", store);
  item.querySelector(".note").addEventListener("input", store);
});
function download() {
  var who = document.getElementById("who").value;
  if (!who) { alert("Pick who you are first."); return; }
  var data = load(), lines = ["episode_id,task,onset,no_clear_moment,note"];
  document.querySelectorAll(".item").forEach(function (item) {
    if (!mine(item, who)) return;
    var e = data[item.dataset.id] || {};
    lines.push([item.dataset.id, item.dataset.task, e.onset || "", e.none ? "yes" : "",
                '"' + (e.note || "").replace(/"/g, '""') + '"'].join(","));
  });
  var link = document.createElement("a");
  link.href = URL.createObjectURL(new Blob([lines.join("\\n") + "\\n"], {type: "text/csv"}));
  link.download = "onset_" + who + ".csv";
  link.click();
}
show();
</script></body></html>
"""


def sheet_image(frames):
    """4 x 4 grid of the 16 pictures with the picture number in the corner."""
    height, width = frames.shape[1:3]
    tile_height = int(round(height * TILE_WIDTH / width))
    sheet = Image.new("RGB", (4 * TILE_WIDTH, 4 * tile_height), "white")
    draw = ImageDraw.Draw(sheet)
    for index, frame in enumerate(frames):
        tile = Image.fromarray(frame).resize((TILE_WIDTH, tile_height))
        x = (index % 4) * TILE_WIDTH
        y = (index // 4) * tile_height
        sheet.paste(tile, (x, y))
        draw.rectangle([x, y, x + 28, y + 18], fill="black")
        draw.text((x + 4, y + 3), str(index), fill="white")
    return sheet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=os.path.dirname(HERE), help="repo folder that holds week2/")
    parser.add_argument("--per-task", type=int, default=20)
    parser.add_argument("--both-per-task", type=int, default=5)
    args = parser.parse_args()

    sys.path.insert(0, os.path.join(args.repo, "week2"))
    from loader import load_episode

    with open(os.path.join(args.repo, "week2", "episode_index.csv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    out = os.path.join(HERE, "onset")
    sheets = os.path.join(out, "sheets")
    os.makedirs(sheets, exist_ok=True)
    rng = random.Random(0)
    table = []
    for task in ("open_oven", "stack_cups"):
        failures = [row for row in rows if row["task"] == task and row["label"] == "failure" and row["split"] == "test"]
        chosen = rng.sample(failures, min(args.per_task, len(failures)))
        for number, row in enumerate(chosen):
            if number < args.both_per_task:
                who = "both"
            else:
                who = "annotator1" if number % 2 == 0 else "annotator2"
            episode = load_episode(row["h5_path"], task)
            # Short names; the episode id is in onset_sheets.csv next to the sheet name.
            name = "%s_%02d.jpg" % ("oven" if task == "open_oven" else "cups", number + 1)
            sheet_image(episode["frames"]).save(os.path.join(sheets, name), quality=90)
            table.append({"task": task, "episode_id": row["episode_id"], "who_labels": who, "sheet": "sheets/" + name})
            print("%s %s %s" % (task, who, row["episode_id"]))

    with open(os.path.join(out, "onset_sheets.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)

    parts = [PAGE_TOP]
    for row in table:
        ident = html.escape(row["episode_id"])
        parts.append('<div class="item" data-id="%s" data-task="%s" data-who="%s" style="display:none">' % (
            ident, row["task"], row["who_labels"]))
        parts.append("<h3>%s (%s)</h3><p>%s<br>marked by: %s</p>" % (
            os.path.basename(row["sheet"]), row["task"], ident, row["who_labels"]))
        parts.append('<img src="%s" loading="lazy" alt="16 frames from %s">' % (row["sheet"], ident))
        parts.append('<p><label>First frame where it went wrong: <input class="onset" type="text" size="3"></label>'
                     '<label><input class="none" type="checkbox"> No clear moment</label>'
                     '<input class="note" type="text" size="40" placeholder="note"> '
                     '<span class="error" style="color:#b00020;font-size:13px"></span></p>')
        parts.append("</div>")
    parts.append(PAGE_BOTTOM)
    page = os.path.join(out, "mark_onsets.html")
    with open(page, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    print("Wrote %d sheets, onset_sheets.csv, and %s" % (len(table), page))


if __name__ == "__main__":
    main()
