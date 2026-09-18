"""
harness.py — everything the 2x2 VLM experiment needs, in one file.

The notebook (notebooks/vlm_2x2_eval.ipynb) holds the config constants and
calls these functions in order. Nothing in here is clever on purpose: plain
functions, plain loops, plain dicts. Read top to bottom.

Sections:
  1. Intake        — frames or video -> 512x512 letterboxed JPEGs
  2. Labels        — import the shirt-colour labels, tag boxes, write manifest.json
  3. Servers       — build llama.cpp, start/health-check the two llama-servers
  4. Querying      — one request to /v1/chat/completions, parse + validate the answer
  5. Running       — run one experiment cell over the manifest, resumable
  6. Metrics       — the six numbers per cell, markdown + CSV table
  7. Error analysis — worst FPs, worst IoU, malformed dump, findings summary
"""

import base64
import csv
import glob
import json
import os
import re
import statistics
import subprocess
import time

import cv2
import numpy as np
import requests


# ============================================================================
# 1. INTAKE
# ============================================================================

def letterbox(img, size=512):
    """Scale the image so its longer side is `size`, then pad the short side with
    black bars so the result is size x size. Aspect ratio is preserved.

    Why not just cv2.resize to 512x512? Stretching a 16:9 frame to a square
    makes people ~1.8x too wide, and every bbox the model returns would then be
    in a distorted coordinate system. The deployment model sees 512x512, so we
    give it exactly what it will get on the robot.
    """
    h, w = img.shape[:2]
    scale = size / max(h, w)
    new_w, new_h = round(w * scale), round(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    top = (size - new_h) // 2
    left = (size - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized
    return canvas


def letterbox_folder(src_dir, out_dir, size=512):
    """Letterbox every image in src_dir into out_dir as frame_XXXX.jpg.
    Files are processed in sorted filename order and renamed to frame_0001,
    frame_0002, ... so the frame id is stable no matter what the source was
    called. The original filename is remembered in frame_sources.json so the
    label import can match on it.
    """
    os.makedirs(out_dir, exist_ok=True)
    exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(src_dir, e)))
    files = sorted(files)
    if not files:
        raise FileNotFoundError(f"No images found in {src_dir}")

    sources = {}  # frame_id -> original filename
    for i, path in enumerate(files, start=1):
        img = cv2.imread(path)
        if img is None:
            print(f"  skipping unreadable file: {path}")
            continue
        frame_id = f"frame_{i:04d}"
        cv2.imwrite(os.path.join(out_dir, frame_id + ".jpg"), letterbox(img, size),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        sources[frame_id] = os.path.basename(path)

    with open(os.path.join(out_dir, "frame_sources.json"), "w") as f:
        json.dump(sources, f, indent=1)
    print(f"Letterboxed {len(sources)} images from {src_dir} -> {out_dir}")
    return sources


def letterbox_video(video_path, out_dir, sample_fps=2.0, size=512):
    """Pull frames from a video at roughly `sample_fps` per second, letterbox
    and save them as frame_XXXX.jpg. Nothing fancy: read every frame, keep
    every N-th one.
    """
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video {video_path}")
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(video_fps / sample_fps))
    print(f"Video is {video_fps:.1f} fps; keeping every {step}th frame (~{video_fps/step:.1f} fps)")

    sources = {}
    n_read, n_kept = 0, 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if n_read % step == 0:
            n_kept += 1
            frame_id = f"frame_{n_kept:04d}"
            cv2.imwrite(os.path.join(out_dir, frame_id + ".jpg"), letterbox(img, size),
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            sources[frame_id] = f"{os.path.basename(video_path)}@frame{n_read}"
        n_read += 1
    cap.release()
    with open(os.path.join(out_dir, "frame_sources.json"), "w") as f:
        json.dump(sources, f, indent=1)
    print(f"Saved {n_kept} letterboxed frames from {video_path} -> {out_dir}")
    return sources


def list_frames(frames_dir):
    """Sorted list of frame ids (frame_0001, ...) present in frames_dir."""
    paths = sorted(glob.glob(os.path.join(frames_dir, "frame_*.jpg")))
    return [os.path.splitext(os.path.basename(p))[0] for p in paths]


# ============================================================================
# 2. LABELS  (import -> tag boxes -> manifest)
# ============================================================================
#
# Labels live in data/labels.json while you are tagging: one entry per frame,
#   {"frame_0001": {"label": "target"|"distractor"|"no_person",
#                   "bbox_px": [x1,y1,x2,y2] or null,     # pixels, 512x512 space
#                   "shirt": "red", "distance": "mid"}}   # optional extras
# The tagger saves this file after EVERY frame so you can stop and resume.
# manifest.json is written from it at the end.


def load_labels(labels_path):
    if os.path.exists(labels_path):
        with open(labels_path) as f:
            return json.load(f)
    return {}


def save_labels(labels, labels_path):
    tmp = labels_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(labels, f, indent=1)
    os.replace(tmp, labels_path)  # atomic: a crash mid-write cannot corrupt the file


def import_labels_from_html(html_path, frame_sources, target_shirt):
    """Read the class labels out of labeled_frames.html (the contact sheet that
    was produced when the clip was labelled). Each card looks like:

        <div class="card " data-shirt="white" data-usable="1">
          ... <div class="fn">frame_0001.jpg</div>
          <span class="chip alt">person</span><span class="chip alt">mid</span>

    Returns {frame_id: {"label", "usable", "shirt", "distance"}}. Frames whose
    shirt colour == target_shirt are 'target'; any other person is a
    'distractor'; 'none' is 'no_person'. Culled frames (usable=0) are kept in
    the dict with usable=False so the manifest step can drop them.

    Matching is by original filename via frame_sources (frame_id -> filename).
    """
    with open(html_path, encoding="utf-8") as f:
        html = f.read()
    # strip the embedded thumbnails first so the regex below stays fast/simple
    html = re.sub(r"data:image/[a-z]+;base64,[A-Za-z0-9+/=]+", "", html)

    card_re = re.compile(
        r'<div class="card[^"]*" data-shirt="([^"]*)" data-usable="([^"]*)">'
        r'.*?<div class="fn">([^<]+)</div>(.*?)</div>\s*</div>', re.S)
    by_filename = {}
    for shirt, usable, filename, meta in card_re.findall(html):
        chips = re.findall(r'<span class="chip alt">([^<]+)</span>', meta)
        distance = chips[1] if len(chips) > 1 else None
        if shirt == "none":
            label = "no_person"
        elif shirt == target_shirt:
            label = "target"
        else:
            label = "distractor"
        by_filename[filename.strip()] = {
            "label": label, "usable": usable == "1",
            "shirt": shirt, "distance": distance}

    imported = {}
    for frame_id, filename in frame_sources.items():
        if filename in by_filename:
            imported[frame_id] = by_filename[filename]
    print(f"Imported labels for {len(imported)} of {len(frame_sources)} frames "
          f"({len(by_filename)} cards in HTML)")
    return imported


def import_labels_from_csv(csv_path, frame_sources, target_shirt):
    """Same idea as the HTML import but from a CSV with columns
    filename, shirt, usable (1/0) and optionally distance."""
    by_filename = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            shirt = row["shirt"].strip()
            if shirt == "none":
                label = "no_person"
            elif shirt == target_shirt:
                label = "target"
            else:
                label = "distractor"
            by_filename[row["filename"].strip()] = {
                "label": label, "usable": str(row.get("usable", "1")).strip() == "1",
                "shirt": shirt, "distance": row.get("distance")}
    imported = {}
    for frame_id, filename in frame_sources.items():
        if filename in by_filename:
            imported[frame_id] = by_filename[filename]
    print(f"Imported labels for {len(imported)} of {len(frame_sources)} frames")
    return imported


def merge_imported_labels(labels, imported):
    """Copy imported class labels into labels.json entries, without touching
    boxes you have already drawn."""
    for frame_id, info in imported.items():
        entry = labels.get(frame_id, {})
        entry.setdefault("bbox_px", None)
        entry["label"] = info["label"]
        entry["usable"] = info["usable"]
        entry["shirt"] = info.get("shirt")
        entry["distance"] = info.get("distance")
        # Empty frames and culled frames need no box, so mark them done now.
        # That way the tagger only stops on frames that actually need a box.
        if info["label"] == "no_person" or not info["usable"]:
            entry["_box_confirmed"] = True
            entry["bbox_px"] = None
        labels[frame_id] = entry
    n_todo = sum(1 for e in labels.values() if not e.get("_box_confirmed"))
    print(f"{n_todo} person frames still need a box in the tagger")
    return labels


# --- optional automatic person detection -----------------------------------
# OpenCV ships a HOG + linear-SVM pedestrian detector inside the library, so
# there is nothing to download and it cannot fail in Colab the way a YOLO
# weights fetch can. It is not great (misses people who fill the frame), but it
# is only a starting point that you confirm or correct.

_hog = None


def detect_person_hog(img):
    """Return the largest HOG person box as [x1,y1,x2,y2] pixels, or None."""
    global _hog
    if _hog is None:
        _hog = cv2.HOGDescriptor()
        _hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    rects, _weights = _hog.detectMultiScale(img, winStride=(8, 8), padding=(8, 8), scale=1.05)
    if len(rects) == 0:
        return None
    x, y, w, h = max(rects, key=lambda r: r[2] * r[3])
    H, W = img.shape[:2]
    return [int(max(0, x)), int(max(0, y)), int(min(W, x + w)), int(min(H, y + h))]


def bbox_px_to_norm(bbox_px, size=512):
    """[x1,y1,x2,y2] pixels -> [x_min,y_min,x_max,y_max] fractions 0-1."""
    if bbox_px is None:
        return None
    return [round(v / size, 4) for v in bbox_px]


def run_tagger(frames_dir, labels_path, auto_detect=True, size=512):
    """The tagging UI. ipywidgets only; ugly but works in Colab.

    For each frame you see the image (with the current box drawn), radio
    buttons for target / distractor / no_person, and a text field with the box
    in PIXELS as 'x1,y1,x2,y2'. If labels were imported the radio is pre-set.
    If auto_detect is on and there is no saved box yet, HOG pre-fills the text.

    Buttons: 'Save & next' writes labels.json immediately and moves on.
             'No box' clears the box and saves.  'Back' / 'Skip' move around.
    Progress is saved after EVERY frame; re-running this cell resumes at the
    first frame without a saved entry.
    """
    import ipywidgets as w
    from IPython.display import display

    frame_ids = list_frames(frames_dir)
    labels = load_labels(labels_path)

    # resume: start at the first frame you have not confirmed yet. Imported
    # labels give a class but no confirmed box, so they still count as "to do".
    start = 0
    for i, fid in enumerate(frame_ids):
        if not labels.get(fid, {}).get("_box_confirmed"):
            start = i
            break
    state = {"i": start}

    img_w = w.Image(format="jpeg", width=size, height=size)
    radio = w.RadioButtons(options=["target", "distractor", "no_person"], description="Class:")
    box_txt = w.Text(description="Box px:", placeholder="x1,y1,x2,y2 (leave empty for no box)")
    status = w.HTML()
    b_save = w.Button(description="Save & next", button_style="success")
    b_nobox = w.Button(description="Save with NO box", button_style="warning")
    b_back = w.Button(description="Back")
    b_skip = w.Button(description="Skip")
    b_hog = w.Button(description="Re-run HOG")

    def parse_box(text):
        text = text.strip()
        if not text:
            return None
        vals = [int(float(v)) for v in text.split(",")]
        if len(vals) != 4 or vals[0] >= vals[2] or vals[1] >= vals[3]:
            raise ValueError("box must be x1,y1,x2,y2 with x1<x2 and y1<y2")
        return [max(0, vals[0]), max(0, vals[1]), min(size, vals[2]), min(size, vals[3])]

    def show(i):
        fid = frame_ids[i]
        entry = labels.get(fid, {})
        img = cv2.imread(os.path.join(frames_dir, fid + ".jpg"))
        box = entry.get("bbox_px")
        if box is None and auto_detect and entry.get("label", "target") != "no_person" and not entry.get("_box_confirmed"):
            box = detect_person_hog(img)
        if box is not None:
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        ok, buf = cv2.imencode(".jpg", img)
        img_w.value = buf.tobytes()
        radio.value = entry.get("label", "target")
        box_txt.value = "" if box is None else ",".join(str(v) for v in box)
        done = sum(1 for f in frame_ids if labels.get(f, {}).get("_box_confirmed"))
        extra = f" · imported: {entry.get('shirt')}/{entry.get('distance')}" if entry.get("shirt") else ""
        status.value = f"<b>{fid}</b>  ({i+1}/{len(frame_ids)}) · confirmed so far: {done}{extra}"

    def save_current(no_box=False):
        fid = frame_ids[state["i"]]
        entry = labels.get(fid, {})
        entry["label"] = radio.value
        if no_box or radio.value == "no_person":
            entry["bbox_px"] = None
        else:
            try:
                entry["bbox_px"] = parse_box(box_txt.value)
            except ValueError as e:
                status.value = f"<span style='color:red'>{e}</span>"
                return False
        entry.setdefault("usable", True)
        entry["_box_confirmed"] = True
        labels[fid] = entry
        save_labels(labels, labels_path)
        return True

    def go(delta):
        state["i"] = max(0, min(len(frame_ids) - 1, state["i"] + delta))
        show(state["i"])

    def go_next_unconfirmed():
        # jump past frames that are already done (imported empty frames etc.)
        for j in range(state["i"] + 1, len(frame_ids)):
            if not labels.get(frame_ids[j], {}).get("_box_confirmed"):
                state["i"] = j
                show(j)
                return
        status.value = "<b>All frames confirmed.</b> Run the manifest cell next. (Use Back/Skip to revisit.)"

    def on_save(_):
        if save_current():
            go_next_unconfirmed()

    def on_nobox(_):
        if save_current(no_box=True):
            go_next_unconfirmed()

    b_save.on_click(on_save)
    b_nobox.on_click(on_nobox)
    b_back.on_click(lambda _: go(-1))
    b_skip.on_click(lambda _: go(+1))

    def on_hog(_):
        img = cv2.imread(os.path.join(frames_dir, frame_ids[state["i"]] + ".jpg"))
        box = detect_person_hog(img)
        box_txt.value = "" if box is None else ",".join(str(v) for v in box)
        status.value = "HOG found nothing — type a box." if box is None else "HOG box loaded."

    b_hog.on_click(on_hog)

    display(w.VBox([status, img_w, radio, box_txt,
                    w.HBox([b_save, b_nobox, b_hog, b_back, b_skip])]))
    show(state["i"])


def write_manifest(frames_dir, labels_path, manifest_path, size=512):
    """labels.json -> data/manifest.json, one record per usable frame:
       {frame_id, path, target_present, distractor_present, bbox, shirt, distance}
    bbox is normalised [x_min, y_min, x_max, y_max] or null.
    Frames with no label at all, or usable=False (culled), are left out.
    """
    labels = load_labels(labels_path)
    records = []
    skipped_unlabelled, skipped_culled = 0, 0
    for fid in list_frames(frames_dir):
        entry = labels.get(fid)
        if not entry or "label" not in entry:
            skipped_unlabelled += 1
            continue
        if not entry.get("usable", True):
            skipped_culled += 1
            continue
        records.append({
            "frame_id": fid,
            "path": os.path.relpath(os.path.join(frames_dir, fid + ".jpg"), os.path.dirname(manifest_path) or "."),
            "target_present": entry["label"] == "target",
            "distractor_present": entry["label"] == "distractor",
            "bbox": bbox_px_to_norm(entry.get("bbox_px"), size),
            "shirt": entry.get("shirt"),
            "distance": entry.get("distance"),
        })
    with open(manifest_path, "w") as f:
        json.dump(records, f, indent=1)
    print(f"Wrote {len(records)} records to {manifest_path} "
          f"(skipped {skipped_unlabelled} unlabelled, {skipped_culled} culled)")
    print_manifest_summary(records)
    return records


def load_manifest(manifest_path):
    with open(manifest_path) as f:
        records = json.load(f)
    # paths in the manifest are relative to the manifest's folder
    base = os.path.dirname(os.path.abspath(manifest_path))
    for r in records:
        if not os.path.isabs(r["path"]):
            r["path"] = os.path.normpath(os.path.join(base, r["path"]))
    return records


def print_manifest_summary(records):
    n_t = sum(r["target_present"] for r in records)
    n_d = sum(r["distractor_present"] for r in records)
    n_n = len(records) - n_t - n_d
    n_box = sum(r["bbox"] is not None for r in records)
    print(f"Frames: {len(records)}  ·  target: {n_t}  ·  distractor: {n_d}  ·  no person: {n_n}  ·  with GT bbox: {n_box}")


# ============================================================================
# 3. SERVERS  (build llama.cpp, verify repos, start, health-check)
# ============================================================================

def build_llama_cpp(llama_dir, cached_bin=None):
    """Clone and build llama.cpp (CPU only) and return the path to llama-server.

    Build order of preference:
      1. already built in llama_dir              -> use it
      2. a copy saved earlier in cached_bin      -> copy it back (Drive survives runtime resets)
      3. build from source (~3-5 min on Colab CPU) and save a copy to cached_bin
    BUILD_SHARED_LIBS=OFF makes llama-server one self-contained file, which is
    what makes caching a single binary possible.
    """
    server_bin = os.path.join(llama_dir, "build", "bin", "llama-server")
    if os.path.exists(server_bin):
        print(f"llama-server already built: {server_bin}")
        return server_bin
    if cached_bin and os.path.exists(cached_bin):
        os.makedirs(os.path.dirname(server_bin), exist_ok=True)
        subprocess.run(["cp", cached_bin, server_bin], check=True)
        os.chmod(server_bin, 0o755)
        print(f"Restored llama-server from cache: {cached_bin}")
        return server_bin

    if not os.path.exists(llama_dir):
        subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/ggml-org/llama.cpp", llama_dir], check=True)
    # -DLLAMA_CURL=ON is what lets the -hf flag download models (needs libcurl dev headers).
    subprocess.run(["cmake", "-B", "build", "-DLLAMA_CURL=ON", "-DGGML_NATIVE=ON",
                    "-DBUILD_SHARED_LIBS=OFF"], cwd=llama_dir, check=True)
    subprocess.run(["cmake", "--build", "build", "--config", "Release", "-j",
                    "--target", "llama-server", "llama-mtmd-cli"], cwd=llama_dir, check=True)
    print(f"Built {server_bin}")
    if cached_bin:
        os.makedirs(os.path.dirname(cached_bin), exist_ok=True)
        subprocess.run(["cp", server_bin, cached_bin], check=True)
        print(f"Saved a copy to {cached_bin}")
    return server_bin


def verify_hf_repo(repo, quant):
    """Ask the Hugging Face API which files the repo has and check that a
    <quant>.gguf and an mmproj-*.gguf are both there. Returns (ok, message).
    Done up front so a missing Model B fails here, with a readable message,
    instead of 40 minutes into the run.
    """
    url = f"https://huggingface.co/api/models/{repo}"
    try:
        r = requests.get(url, timeout=30)
    except requests.RequestException as e:
        return False, f"{repo}: could not reach Hugging Face ({e})"
    if r.status_code != 200:
        return False, f"{repo}: HTTP {r.status_code} from Hugging Face — repo missing or private?"
    files = [s["rfilename"] for s in r.json().get("siblings", [])]
    model_files = [f for f in files if f.lower().endswith(".gguf")
                   and quant.lower() in f.lower() and not f.lower().startswith("mmproj")]
    mmproj_files = [f for f in files if f.lower().startswith("mmproj") and f.lower().endswith(".gguf")]
    if not model_files:
        return False, f"{repo}: no {quant} .gguf found. Files: {files}"
    if not mmproj_files:
        return False, f"{repo}: no mmproj-*.gguf found (vision projector) — cannot do images."
    # prefer the mmproj with the same quant, else the first one
    same_quant = [f for f in mmproj_files if quant.lower() in f.lower()]
    mmproj = (same_quant or mmproj_files)[0]
    return True, f"{repo}: ok — {model_files[0]} + {mmproj}  (mmproj file: {mmproj})"


def start_server(server_bin, hf_repo, quant, port, cache_dir, log_path, threads=None, mmproj_file=None):
    """Start llama-server in the background and return the Popen handle.

    `-hf repo:quant` makes llama.cpp download the language GGUF AND the matching
    mmproj automatically (do not fetch mmproj by hand). LLAMA_CACHE points the
    download at Google Drive so a disconnect does not force a re-download.

    mmproj_file is an escape hatch: if the smoke test says the server cannot
    take images, pass the mmproj filename from verify_hf_repo() and it is
    downloaded explicitly with --mmproj-url instead of relying on auto-detect.
    """
    os.makedirs(cache_dir, exist_ok=True)
    env = dict(os.environ, LLAMA_CACHE=cache_dir)
    cmd = [server_bin, "-hf", f"{hf_repo}:{quant}", "--port", str(port), "--host", "127.0.0.1"]
    if mmproj_file:
        cmd += ["--mmproj-url", f"https://huggingface.co/{hf_repo}/resolve/main/{mmproj_file}"]
    if threads:
        cmd += ["-t", str(threads)]
    log = open(log_path, "a")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
    print(f"Started {' '.join(cmd)}  (pid {proc.pid}, log {log_path})")
    return proc


def wait_for_health(port, timeout_s=1800, log_path=None):
    """Poll /health until the server says ok. Long timeout because the first
    call includes the model download. Returns True/False, never raises."""
    url = f"http://127.0.0.1:{port}/health"
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200 and r.json().get("status") == "ok":
                print(f"port {port}: healthy after {time.time()-t0:.0f}s")
                return True
        except requests.RequestException:
            pass
        time.sleep(5)
    print(f"port {port}: NOT healthy after {timeout_s}s")
    if log_path and os.path.exists(log_path):
        with open(log_path) as f:
            print("--- last lines of server log ---")
            print("".join(f.readlines()[-25:]))
    return False


def peak_rss_mb(pid):
    """Peak resident memory of a process, from the kernel's own high-water mark
    (VmHWM in /proc/<pid>/status). No extra library needed. None if gone."""
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) / 1024.0
    except FileNotFoundError:
        return None
    return None


# ============================================================================
# 4. QUERYING  (one request, parse, validate)
# ============================================================================

def image_to_data_url(path):
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return "data:image/jpeg;base64," + b64


def query_model(port, prompt_text, image_path, grammar=None, max_tokens=128, seed=42, timeout_s=180):
    """One call to llama-server's OpenAI-compatible endpoint.
    Returns (raw_text, latency_ms, server_timings, error). On any network/HTTP
    problem raw_text is '' and error explains why — the caller decides what to do.

    temperature=0 and a fixed seed make the run repeatable. `grammar` is a
    llama.cpp extension to the OpenAI schema; when set, decoding is constrained
    to grammars/bbox.gbnf. (Some older builds only accept `json_schema`;
    if you get HTTP 400 mentioning grammar, that is the reason.)
    """
    body = {
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": image_to_data_url(image_path)}},
            ],
        }],
        "temperature": 0,
        "seed": seed,
        "max_tokens": max_tokens,
    }
    if grammar:
        body["grammar"] = grammar
    t0 = time.perf_counter()
    try:
        r = requests.post(f"http://127.0.0.1:{port}/v1/chat/completions", json=body, timeout=timeout_s)
    except requests.RequestException as e:
        return "", (time.perf_counter() - t0) * 1000, None, f"request failed: {type(e).__name__}"
    latency_ms = (time.perf_counter() - t0) * 1000
    if r.status_code != 200:
        return "", latency_ms, None, f"HTTP {r.status_code}: {r.text[:200]}"
    data = r.json()
    try:
        raw = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return "", latency_ms, None, f"unexpected response shape: {str(data)[:200]}"
    return raw, latency_ms, data.get("timings"), None


def parse_answer(raw):
    """Best-effort parse of the model's text into {"match", "confidence", "bbox"}.
    Returns (parsed_dict_or_None, reason). reason is '' when valid, otherwise a
    short string saying what was wrong — this is what the malformed-output
    metric counts.

    'Best effort' means: find the first {...} block anywhere in the text (free
    generation likes to wrap JSON in prose or ```json fences), then check the
    three keys have the right types and ranges. Anything else is malformed.
    """
    if raw is None or not raw.strip():
        return None, "empty response"
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return None, "no JSON object found"
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None, "JSON does not parse"
    if not isinstance(obj, dict):
        return None, "JSON is not an object"
    if not isinstance(obj.get("match"), bool):
        return None, "match missing or not a bool"
    conf = obj.get("confidence")
    if not isinstance(conf, (int, float)) or isinstance(conf, bool) or not (0 <= conf <= 1):
        return None, "confidence missing or not a number in [0,1]"
    bbox = obj.get("bbox", "MISSING")
    if bbox == "MISSING":
        return None, "bbox key missing"
    if bbox is not None:
        if (not isinstance(bbox, list) or len(bbox) != 4
                or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in bbox)
                or not all(0 <= v <= 1 for v in bbox)):
            return None, "bbox not null or 4 numbers in [0,1]"
        if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
            return None, "bbox has zero/negative size"
    return {"match": obj["match"], "confidence": float(conf), "bbox": bbox}, ""


def iou(a, b):
    """Intersection-over-union of two [x_min,y_min,x_max,y_max] boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# ============================================================================
# 5. RUNNING one experiment cell
# ============================================================================

def load_jsonl(path):
    rows = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def run_cell(cell_name, port, server_pid, prompt_text, grammar_text, manifest,
             raw_dir, max_tokens=128, seed=42, progress_every=10, max_consecutive_errors=5):
    """Run every manifest frame through one model+prompt combination.

    Results go to raw_dir/<cell>.jsonl, ONE LINE PER FRAME, appended as soon as
    each answer comes back. If Colab dies, re-running this function skips the
    frames already in the file — that is the whole resume mechanism.

    Each line: frame_id, raw (exact model text), parsed, malformed_reason,
    correct (match == target_present, None if malformed), iou (or None),
    latency_ms, server_timings, error.

    Stops early (partial results kept) after `max_consecutive_errors` failed
    requests in a row, which almost always means the server died.
    """
    os.makedirs(raw_dir, exist_ok=True)
    out_path = os.path.join(raw_dir, f"{cell_name}.jsonl")
    meta_path = os.path.join(raw_dir, f"{cell_name}_meta.json")

    done_ids = {r["frame_id"] for r in load_jsonl(out_path)}
    todo = [r for r in manifest if r["frame_id"] not in done_ids]
    print(f"[{cell_name}] {len(done_ids)} frames already done, {len(todo)} to go")
    if not todo:
        return

    consecutive_errors = 0
    t_start = time.time()
    with open(out_path, "a") as out:
        for n, rec in enumerate(todo, start=1):
            raw, latency_ms, timings, error = query_model(
                port, prompt_text, rec["path"], grammar=grammar_text,
                max_tokens=max_tokens, seed=seed)
            if error:
                consecutive_errors += 1
                parsed, reason = None, f"request error: {error}"
            else:
                consecutive_errors = 0
                parsed, reason = parse_answer(raw)

            correct = None
            frame_iou = None
            if parsed is not None:
                correct = parsed["match"] == rec["target_present"]
                if parsed["bbox"] is not None and rec["bbox"] is not None:
                    frame_iou = iou(parsed["bbox"], rec["bbox"])

            out.write(json.dumps({
                "frame_id": rec["frame_id"], "raw": raw, "parsed": parsed,
                "malformed_reason": reason, "correct": correct, "iou": frame_iou,
                "latency_ms": round(latency_ms, 1), "server_timings": timings, "error": error,
            }) + "\n")
            out.flush()

            if n % progress_every == 0 or n == len(todo):
                elapsed = time.time() - t_start
                eta = elapsed / n * (len(todo) - n)
                print(f"[{cell_name}] {n}/{len(todo)}  last {latency_ms:.0f} ms  "
                      f"elapsed {elapsed/60:.1f} min  eta {eta/60:.1f} min")
            if consecutive_errors >= max_consecutive_errors:
                print(f"[{cell_name}] STOPPING: {consecutive_errors} requests failed in a row "
                      f"(last: {error}). Is the server on port {port} still alive? "
                      f"Partial results are saved; re-run this cell to resume.")
                break

    # Peak memory is a property of the server process, not of a frame, so it
    # is stored once per cell. Read after the run so it includes inference.
    meta = {"cell": cell_name, "port": port, "server_pid": server_pid,
            "peak_rss_mb": peak_rss_mb(server_pid) if server_pid else None,
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=1)
    print(f"[{cell_name}] done. peak server RSS: {meta['peak_rss_mb']} MB")


# ============================================================================
# 6. METRICS  (six numbers per cell -> markdown + CSV)
# ============================================================================

def percentile(values, p):
    """p-th percentile without numpy ceremony (nearest-rank)."""
    if not values:
        return None
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round(p / 100.0 * len(s) + 0.5)) - 1))
    return s[k]


def compute_metrics(cell_name, raw_dir, manifest, malformed_policy="count_as_miss"):
    """The six metrics for one cell, as a plain dict. None where undefined.

    malformed_policy decides what a malformed answer means for accuracy/FPR:
      'count_as_miss' — the robot got no usable answer, so it is scored as WRONG
                        for that frame (for a distractor frame that means it
                        counts as a false positive). Default, and the stricter
                        choice: in practice an unparseable answer is a failure.
      'exclude'       — the frame is dropped from accuracy/FPR denominators, so
                        those two numbers describe only the well-formed answers.
    Either way the malformed-output rate itself is always reported over all
    frames, so nothing is hidden.

    Frames whose request failed outright (server down, HTTP error) are a
    harness problem, not a model answer; they are excluded from everything and
    reported in 'n_request_errors'.
    """
    rows = load_jsonl(os.path.join(raw_dir, f"{cell_name}.jsonl"))
    by_id = {r["frame_id"]: r for r in manifest}
    meta_path = os.path.join(raw_dir, f"{cell_name}_meta.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}

    answered = [r for r in rows if not r["error"]]
    n_errors = len(rows) - len(answered)
    malformed = [r for r in answered if r["parsed"] is None]

    # --- 1. match accuracy
    acc_hits, acc_total = 0, 0
    # --- 2. false-positive rate on distractor frames
    fp_hits, fp_total = 0, 0
    for r in answered:
        gt = by_id[r["frame_id"]]
        is_malformed = r["parsed"] is None
        if is_malformed and malformed_policy == "exclude":
            continue
        acc_total += 1
        if not is_malformed and r["correct"]:
            acc_hits += 1
        if gt["distractor_present"]:
            fp_total += 1
            if is_malformed or r["parsed"]["match"]:
                fp_hits += 1

    # --- 3. IoU, only where both boxes exist
    ious = [r["iou"] for r in answered if r["iou"] is not None]
    # --- 5. latency
    lat = [r["latency_ms"] for r in answered]

    return {
        "cell": cell_name,
        "n_frames": len(rows),
        "n_request_errors": n_errors,
        "match_accuracy": acc_hits / acc_total if acc_total else None,
        "false_positive_rate": fp_hits / fp_total if fp_total else None,
        "n_distractor": fp_total,
        "mean_iou": statistics.mean(ious) if ious else None,
        "n_iou": len(ious),
        "malformed_rate": len(malformed) / len(answered) if answered else None,
        "n_malformed": len(malformed),
        "latency_mean_ms": statistics.mean(lat) if lat else None,
        "latency_p95_ms": percentile(lat, 95),
        "peak_rss_mb": meta.get("peak_rss_mb"),
    }


def _fmt(v, kind):
    if v is None:
        return "n/a"
    if kind == "pct":
        return f"{100*v:.1f}%"
    if kind == "f3":
        return f"{v:.3f}"
    if kind == "ms":
        return f"{v:.0f}"
    if kind == "mb":
        return f"{v:.0f}"
    return str(v)


def write_metrics_table(metrics_list, cells_info, results_dir, malformed_policy):
    """metrics_list: output of compute_metrics for each cell (in A1,A2,B1,B2 order).
    cells_info: {cell: {"model": ..., "prompt": ..., "grammar": bool}} for labels.
    Writes results/metrics.md and results/metrics.csv and returns the markdown."""
    os.makedirs(results_dir, exist_ok=True)
    lines = []
    lines.append("# 2x2 results\n")
    lines.append(f"Malformed answers in free-generation cells are scored with policy **`{malformed_policy}`** "
                 + ("(a malformed answer counts as a wrong answer; on a distractor frame it counts as a false positive)."
                    if malformed_policy == "count_as_miss" else
                    "(malformed answers are dropped from the accuracy and false-positive denominators).") + "\n")
    lines.append("| Cell | Model | Prompt | Match acc. | False-pos. rate | Mean IoU (n) | Malformed rate | Latency mean / p95 ms | Peak RSS MB |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for m in metrics_list:
        info = cells_info[m["cell"]]
        prompt_label = info["prompt"] + (" + grammar" if info["grammar"] else "")
        lines.append(
            f"| {m['cell']} | {info['model']} | {prompt_label} | {_fmt(m['match_accuracy'],'pct')} | "
            f"{_fmt(m['false_positive_rate'],'pct')} (n={m['n_distractor']}) | "
            f"{_fmt(m['mean_iou'],'f3')} ({m['n_iou']}) | {_fmt(m['malformed_rate'],'pct')} ({m['n_malformed']}) | "
            f"{_fmt(m['latency_mean_ms'],'ms')} / {_fmt(m['latency_p95_ms'],'ms')} | {_fmt(m['peak_rss_mb'],'mb')} |")
    n_err = sum(m["n_request_errors"] for m in metrics_list)
    if n_err:
        lines.append(f"\n**Warning:** {n_err} request(s) failed outright (server down / HTTP error) and are excluded from all metrics. "
                     "Re-run the affected cell to fill them in.\n")
    md = "\n".join(lines) + "\n"
    with open(os.path.join(results_dir, "metrics.md"), "w") as f:
        f.write(md)

    csv_fields = ["cell", "model", "prompt", "grammar", "n_frames", "n_request_errors", "match_accuracy",
                  "false_positive_rate", "n_distractor", "mean_iou", "n_iou", "malformed_rate", "n_malformed",
                  "latency_mean_ms", "latency_p95_ms", "peak_rss_mb", "malformed_policy"]
    with open(os.path.join(results_dir, "metrics.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=csv_fields)
        wr.writeheader()
        for m in metrics_list:
            row = dict(m)
            row.update(cells_info[m["cell"]])
            row["malformed_policy"] = malformed_policy
            wr.writerow({k: row.get(k) for k in csv_fields})
    return md


# ============================================================================
# 7. ERROR ANALYSIS + findings summary
# ============================================================================

def _delta(a, b):
    """b - a, or None if either is missing."""
    if a is None or b is None:
        return None
    return b - a


def _pp(v):
    """format a delta of a rate in percentage points"""
    return "n/a" if v is None else f"{100*v:+.1f} pp"


def findings_summary(metrics, cells_info):
    """Plain-English findings: the two refinement deltas, the interaction, and a
    verdict on which model to deploy. `metrics` is {cell: metrics_dict}.

    Verdict rule (stated here so it can be defended): compare the two
    grammar-constrained cells A2 vs B2, because that is what would actually be
    deployed. Lowest false-positive rate wins; if the FPRs are within 2
    percentage points it is a tie on FPR and match accuracy decides; if that is
    within 2 pp too, mean IoU decides; latency is reported but only breaks a
    full tie. FPR comes first because a distractor mistaken for the target
    makes the robot follow the wrong person, which is the failure that matters.
    """
    A1, A2, B1, B2 = (metrics.get(c) for c in ("A1", "A2", "B1", "B2"))
    model_a = cells_info["A1"]["model"]
    model_b = cells_info["B1"]["model"]
    out = ["# Findings\n"]

    def block(name, m1, m2):
        if m1 is None or m2 is None:
            return f"**{name}:** one of the cells has no results yet; delta not computed.\n"
        return (f"**{name} — v1 → v2 + grammar:** "
                f"FPR {_fmt(m1['false_positive_rate'],'pct')} → {_fmt(m2['false_positive_rate'],'pct')} "
                f"({_pp(_delta(m1['false_positive_rate'], m2['false_positive_rate']))}); "
                f"accuracy {_fmt(m1['match_accuracy'],'pct')} → {_fmt(m2['match_accuracy'],'pct')} "
                f"({_pp(_delta(m1['match_accuracy'], m2['match_accuracy']))}); "
                f"malformed {_fmt(m1['malformed_rate'],'pct')} → {_fmt(m2['malformed_rate'],'pct')}; "
                f"IoU {_fmt(m1['mean_iou'],'f3')} → {_fmt(m2['mean_iou'],'f3')}.\n")

    out.append(block(model_a, A1, A2))
    out.append(block(model_b, B1, B2))

    # --- interaction: did refinement help both models equally?
    if all(m is not None for m in (A1, A2, B1, B2)):
        dA_fpr = _delta(A1["false_positive_rate"], A2["false_positive_rate"])
        dB_fpr = _delta(B1["false_positive_rate"], B2["false_positive_rate"])
        dA_acc = _delta(A1["match_accuracy"], A2["match_accuracy"])
        dB_acc = _delta(B1["match_accuracy"], B2["match_accuracy"])
        out.append("## Interaction (model x prompt)\n")
        out.append(f"Refinement changed FPR by {_pp(dA_fpr)} for {model_a} and {_pp(dB_fpr)} for {model_b}; "
                   f"accuracy by {_pp(dA_acc)} and {_pp(dB_acc)}.\n")
        if dA_acc is not None and dB_acc is not None:
            gap = abs(dA_acc - dB_acc)
            weaker = model_a if (A1["match_accuracy"] or 0) < (B1["match_accuracy"] or 0) else model_b
            weaker_gain = dA_acc if weaker == model_a else dB_acc
            stronger_gain = dB_acc if weaker == model_a else dA_acc
            if gap < 0.05:
                out.append("The two accuracy deltas are within 5 pp of each other: **refinement helped both models "
                           "about equally** (no meaningful interaction).\n")
            elif weaker_gain > stronger_gain:
                out.append(f"**Interaction present:** refinement mainly rescued the weaker baseline ({weaker}), "
                           f"which gained {_pp(weaker_gain)} vs {_pp(stronger_gain)} for the other model. "
                           "The baseline model ranking is therefore not the ranking that matters for deployment.\n")
            else:
                out.append(f"**Interaction present:** refinement helped the stronger baseline more "
                           f"({_pp(stronger_gain)} vs {_pp(weaker_gain)} for {weaker}), widening the gap.\n")

    # --- verdict
    out.append("## Which model to deploy\n")
    if A2 is None or B2 is None:
        have = [c for c, m in (("A2", A2), ("B2", B2)) if m is not None]
        out.append(f"Only {have or 'neither'} of the grammar cells has results, so no head-to-head verdict yet. "
                   "Run the missing cell (Model B may have failed to start — check the server log).\n")
    else:
        reasons = []
        winner = None
        fa, fb = A2["false_positive_rate"], B2["false_positive_rate"]
        aa, ab = A2["match_accuracy"], B2["match_accuracy"]
        ia, ib = A2["mean_iou"], B2["mean_iou"]
        if fa is not None and fb is not None and abs(fa - fb) > 0.02:
            winner = model_a if fa < fb else model_b
            reasons.append(f"lower false-positive rate ({_fmt(min(fa,fb),'pct')} vs {_fmt(max(fa,fb),'pct')})")
        elif aa is not None and ab is not None and abs(aa - ab) > 0.02:
            winner = model_a if aa > ab else model_b
            reasons.append(f"FPR is a tie (within 2 pp), higher match accuracy ({_fmt(max(aa,ab),'pct')} vs {_fmt(min(aa,ab),'pct')})")
        elif ia is not None and ib is not None and abs(ia - ib) > 0.02:
            winner = model_a if ia > ib else model_b
            reasons.append(f"FPR and accuracy tie, better IoU ({_fmt(max(ia,ib),'f3')} vs {_fmt(min(ia,ib),'f3')})")
        else:
            la, lb = A2["latency_mean_ms"], B2["latency_mean_ms"]
            if la is not None and lb is not None and la != lb:
                winner = model_a if la < lb else model_b
                reasons.append(f"quality metrics all tie, lower latency ({_fmt(min(la,lb),'ms')} vs {_fmt(max(la,lb),'ms')} ms)")
        if winner is None:
            out.append("**No clear winner** — A2 and B2 tie on every metric within the thresholds. Either model is defensible; "
                       "pick the one with lower latency/memory on the Pi.\n")
        else:
            other = model_b if winner == model_a else model_a
            out.append(f"**Deploy {winner}** (with the v2 prompt + grammar). Reason: {reasons[0]}. "
                       f"Latency {_fmt(A2['latency_mean_ms'],'ms')} ms ({model_a}) vs {_fmt(B2['latency_mean_ms'],'ms')} ms ({model_b}); "
                       f"peak RSS {_fmt(A2['peak_rss_mb'],'mb')} vs {_fmt(B2['peak_rss_mb'],'mb')} MB. "
                       f"Runner-up: {other}.\n")
        out.append("Rule used: FPR first (a wrong-person follow is the failure that matters), then accuracy, then IoU, "
                   "then latency; differences under 2 pp / 0.02 are treated as ties. Latency here is Colab CPU, "
                   "not the Pi 5 — only the ratio between models is meaningful.\n")
    return "\n".join(out)


def write_error_analysis(cells, raw_dir, manifest, results_dir, metrics, cells_info, top_n=10):
    """results/error_analysis.md: per cell, the worst false positives (highest
    confidence first), false alarms on empty frames, worst IoU misses, and every
    malformed output with its raw text — each with the frame path so you can
    open the image. Ends with the findings summary."""
    # show frame paths relative to the project folder so they are short and clickable
    project_dir = os.path.dirname(os.path.abspath(results_dir))
    by_id = {}
    for r in manifest:
        r = dict(r)
        r["path"] = os.path.relpath(r["path"], project_dir)
        by_id[r["frame_id"]] = r
    out = ["# Error analysis\n"]
    for cell in cells:
        rows = [r for r in load_jsonl(os.path.join(raw_dir, f"{cell}.jsonl")) if not r["error"]]
        if not rows:
            out.append(f"## {cell}\n\nNo results.\n")
            continue
        out.append(f"## {cell} — {cells_info[cell]['model']}, {cells_info[cell]['prompt']}"
                   f"{' + grammar' if cells_info[cell]['grammar'] else ''}\n")

        fps = [r for r in rows if r["parsed"] and r["parsed"]["match"] and by_id[r["frame_id"]]["distractor_present"]]
        fps.sort(key=lambda r: -r["parsed"]["confidence"])
        out.append(f"### Worst false positives (distractor called target) — {len(fps)} total, top {min(top_n, len(fps))}\n")
        if fps:
            out.append("| frame | shirt / distance | confidence | pred bbox | raw |")
            out.append("|---|---|---|---|---|")
            for r in fps[:top_n]:
                g = by_id[r["frame_id"]]
                out.append(f"| `{g['path']}` | {g.get('shirt')} / {g.get('distance')} | {r['parsed']['confidence']:.2f} | "
                           f"{r['parsed']['bbox']} | `{r['raw'].strip()[:80].replace('|','/').replace(chr(10),' ')}` |")
        out.append("")

        empties = [r for r in rows if r["parsed"] and r["parsed"]["match"]
                   and not by_id[r["frame_id"]]["target_present"] and not by_id[r["frame_id"]]["distractor_present"]]
        out.append(f"### False alarms on empty frames — {len(empties)}\n")
        if empties:
            out.append(", ".join(f"`{by_id[r['frame_id']]['path']}` ({r['parsed']['confidence']:.2f})" for r in empties[:top_n]) + "\n")

        misses = [r for r in rows if r["parsed"] and by_id[r["frame_id"]]["target_present"]
                  and not r["parsed"]["match"]]
        out.append(f"### Missed targets (target present, match=false) — {len(misses)}\n")
        if misses:
            out.append(", ".join(f"`{by_id[r['frame_id']]['path']}`" for r in misses[:top_n]) + "\n")

        ious = [r for r in rows if r["iou"] is not None]
        ious.sort(key=lambda r: r["iou"])
        out.append(f"### Worst IoU — {len(ious)} frames with both boxes, lowest {min(top_n, len(ious))}\n")
        if ious:
            out.append("| frame | IoU | GT bbox | pred bbox |")
            out.append("|---|---|---|---|")
            for r in ious[:top_n]:
                g = by_id[r["frame_id"]]
                out.append(f"| `{g['path']}` | {r['iou']:.3f} | {g['bbox']} | {r['parsed']['bbox']} |")
        out.append("")

        bad = [r for r in rows if r["parsed"] is None]
        out.append(f"### Malformed outputs — {len(bad)}\n")
        for r in bad:
            g = by_id[r["frame_id"]]
            raw = r["raw"].strip().replace("```", "'''")
            out.append(f"- `{g['path']}` — {r['malformed_reason']}\n\n  ```\n  {raw[:400]}\n  ```")
        out.append("")

    out.append(findings_summary(metrics, cells_info))
    md = "\n".join(out)
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "error_analysis.md"), "w") as f:
        f.write(md)
    return md
