"""AutoHDR web app — upload bracketed exposures, get the enhanced result.

Run:  python3 scripts/app.py   then open http://127.0.0.1:5050
"""
import io
import os
import shutil
import tempfile
import time
from pathlib import Path


def _ensure_temp_space(min_free_gb=2):
    """Uploads spool to the OS temp dir; if its drive is nearly full
    (e.g. a packed C:), fall back to another drive so multi-bracket
    uploads don't fail with ENOSPC."""
    if shutil.disk_usage(tempfile.gettempdir()).free >= min_free_gb * 2**30:
        return
    for letter in "DEFGH":
        root = f"{letter}:\\"
        if os.path.exists(root) and \
                shutil.disk_usage(root).free >= min_free_gb * 2**30:
            t = os.path.join(root, "autohdr-tmp")
            os.makedirs(t, exist_ok=True)
            tempfile.tempdir = t
            os.environ["TMP"] = os.environ["TEMP"] = t
            os.environ.setdefault("HF_HOME", os.path.join(t, "hf-cache"))
            print(f" * temp dir relocated to {t} (system temp drive is full)")
            return


_ensure_temp_space()

import cv2
import numpy as np
from flask import Flask, request, send_file, render_template_string

import pipeline

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "web_output"
if shutil.disk_usage(ROOT).free < 1 * 2**30:
    # project drive can't hold ~25MB results — save next to the temp dir
    RESULTS = Path(tempfile.gettempdir()) / "web_output"
    print(f" * results relocated to {RESULTS} (project drive is full)")
RESULTS.mkdir(exist_ok=True)

app = Flask(__name__)

PAGE = """
<!doctype html>
<title>AutoHDR — bracket merge</title>
<style>
  body { font-family: -apple-system, sans-serif; max-width: 720px; margin: 40px auto; padding: 0 16px; }
  .card { border: 1px solid #ddd; border-radius: 12px; padding: 24px; }
  button { background: #2563eb; color: white; border: 0; border-radius: 8px; padding: 10px 24px; font-size: 15px; cursor: pointer; }
  img { max-width: 100%; border-radius: 8px; margin-top: 16px; }
  .err { color: #b91c1c; }
</style>
<h1>AutoHDR</h1>
<div class="card">
  <p>Same scene 2–5 different exposure photos select:</p>
  <form method="post" action="/process" enctype="multipart/form-data">
    <input type="file" name="brackets" accept="image/*" multiple required>
    <p><button>Merge & Enhance</button></p>
  </form>
  {% if error %}<p class="err">{{ error }}</p>{% endif %}
  {% if result %}
    <h3>Result ({{ took }}s)</h3>
    <img src="/result/{{ result }}">
    <p><a href="/result/{{ result }}" download>Download JPEG</a></p>
  {% endif %}
</div>
"""


@app.get("/")
def index():
    return render_template_string(PAGE)


@app.post("/process")
def process():
    files = request.files.getlist("brackets")
    images = []
    for f in files:
        data = np.frombuffer(f.read(), np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is not None:
            images.append(img)
    if len(images) < 2:
        return render_template_string(
            PAGE, error="Kam se kam 2 valid images chahiye (jpg/png)."
        )

    start = time.time()
    result = pipeline.process_brackets(images)
    took = round(time.time() - start, 1)

    name = f"hdr_{int(time.time() * 1000)}.jpg"
    if not cv2.imwrite(str(RESULTS / name), result,
                       [cv2.IMWRITE_JPEG_QUALITY, 95]):
        return render_template_string(
            PAGE, error="Result save failed — disk full? Free some space "
                        f"on {RESULTS.drive or 'the results drive'} and retry."
        )
    return render_template_string(PAGE, result=name, took=took)


@app.get("/result/<name>")
def result(name):
    path = (RESULTS / Path(name).name).resolve()
    if not path.is_file() or path.parent != RESULTS.resolve():
        return "not found", 404
    return send_file(path, mimetype="image/jpeg")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050, debug=False)
