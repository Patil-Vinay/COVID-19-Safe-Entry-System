# Web page for the staff, runs on the Pi next to sentinel.py.
# Open http://<pi address>:5000 on a phone or laptop on the same Wi-Fi.
import csv
import os
from collections import Counter
from datetime import datetime

from flask import Flask, redirect, render_template_string

import config

app = Flask(__name__)

PAGE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="5">
<title>Sentinel - Staff</title>
<style>
  body { font-family: Arial, sans-serif; background: #f2f2f2; margin: 0; padding: 16px; color: #222; }
  h1 { font-size: 22px; margin: 0 0 4px; }
  .sub { color: #777; margin-bottom: 16px; }
  .cards { display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 16px; }
  .card { background: white; border-radius: 8px; padding: 14px 18px; min-width: 110px; flex: 1; }
  .card .label { color: #777; font-size: 14px; }
  .card .value { font-size: 32px; font-weight: bold; margin-top: 4px; }
  .full { color: #d33; }
  .box { background: white; border-radius: 8px; padding: 14px 18px; margin-bottom: 16px; }
  table { width: 100%; border-collapse: collapse; font-size: 14px; }
  th, td { text-align: left; padding: 6px 4px; border-bottom: 1px solid #eee; }
  .allowed { color: #2a8a2a; }
  .exit { color: #555; }
  .refused { color: #d33; }
  button { background: #d33; color: white; border: none; border-radius: 6px; padding: 10px 16px; font-size: 15px; }
</style>
</head>
<body>
<h1>THE SENTINEL</h1>
<div class="sub">Staff page, updates every 5 seconds</div>

<div class="cards">
  <div class="card"><div class="label">People inside</div>
    <div class="value {{ 'full' if inside >= max_people }}">{{ inside }} / {{ max_people }}</div></div>
  <div class="card"><div class="label">Entered today</div><div class="value">{{ entered }}</div></div>
  <div class="card"><div class="label">Left today</div><div class="value">{{ left }}</div></div>
  <div class="card"><div class="label">Refused today</div><div class="value">{{ refused_total }}</div></div>
</div>

{% if refused %}
<div class="box">
  <b>Why people were refused today</b>
  <table>
  {% for reason, count in refused %}
    <tr><td>{{ reason }}</td><td>{{ count }}</td></tr>
  {% endfor %}
  </table>
</div>
{% endif %}

<div class="box">
  <b>Last events today</b>
  <table>
    <tr><th>Time</th><th>Result</th><th>Temp (C)</th><th>No mask score</th><th>Inside</th></tr>
    {% for r in recent %}
    <tr>
      <td>{{ r.time[11:] }}</td>
      <td class="{{ r.css }}">{{ r.result }}</td>
      <td>{{ r.temperature }}</td>
      <td>{{ r.no_mask_score }}</td>
      <td>{{ r.inside }}</td>
    </tr>
    {% else %}
    <tr><td colspan="5">Nothing yet</td></tr>
    {% endfor %}
  </table>
</div>

<form method="post" action="/reset" onsubmit="return confirm('Set the people count to 0?')">
  <button type="submit">Reset count</button>
</form>
</body>
</html>
"""

# results that are not a refused entry
NOT_REFUSED = ("allowed", "exit", "did not enter", "started", "count reset")


def read_log():
    if not os.path.exists(config.LOG_FILE):
        return []
    with open(config.LOG_FILE, newline="") as f:
        return list(csv.DictReader(f))


@app.route("/")
def index():
    rows = read_log()
    today = datetime.now().strftime("%Y-%m-%d")
    today_rows = [r for r in rows if r["time"].startswith(today)]
    refused = Counter(r["result"] for r in today_rows if r["result"] not in NOT_REFUSED)

    recent = []
    for r in reversed(today_rows[-20:]):
        if r["result"] == "allowed":
            r["css"] = "allowed"
        elif r["result"] in NOT_REFUSED:
            r["css"] = "exit"
        else:
            r["css"] = "refused"
        recent.append(r)

    return render_template_string(
        PAGE,
        inside=int(rows[-1]["inside"]) if rows else 0,
        max_people=config.MAX_PEOPLE,
        entered=sum(1 for r in today_rows if r["result"] == "allowed"),
        left=sum(1 for r in today_rows if r["result"] == "exit"),
        refused_total=sum(refused.values()),
        refused=refused.most_common(),
        recent=recent)


@app.route("/reset", methods=["POST"])
def reset():
    # sentinel.py sees this file, sets the count to 0 and deletes it
    os.makedirs(os.path.dirname(config.RESET_FILE), exist_ok=True)
    open(config.RESET_FILE, "w").close()
    return redirect("/")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
