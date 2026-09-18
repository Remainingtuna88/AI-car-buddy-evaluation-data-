# 2×2 VLM evaluation harness

Which small vision-language model should run onboard the MentorPi search-and-locate robot, and
does prompt refinement plus grammar-constrained decoding actually make it more reliable?

The robot's vision loop asks one question of each sampled camera frame — *"is the person in this
image the target, and where are they?"* — and needs a machine-readable answer back:

```json
{"match": true, "confidence": 0.92, "bbox": [0.38, 0.12, 0.61, 0.95]}
```

This harness runs that exact question over a labelled set of frames under four conditions, and
reports six metrics per condition plus an error analysis and a deployment recommendation.

---

## The experiment

|        | **Prompt v1** (weak baseline, free generation) | **Prompt v2** (refined) + GBNF grammar |
|---|---|---|
| **Model A** — LFM2.5-VL-450M | **A1** | **A2** |
| **Model B** — SmolVLM2-500M  | **B1** | **B2** |

- **Model A:** `LiquidAI/LFM2.5-VL-450M-GGUF`, Q8_0 — the incumbent, already named in the
  MentorPi deploy notes.
- **Model B:** `ggml-org/SmolVLM2-500M-Video-Instruct-GGUF`, Q8_0 — the size-matched challenger.

Both run under **one** llama.cpp build on the same CPU, at the same quantization, so the latency
and memory columns are comparable rather than apples-to-oranges. (SmolVLM2's GGUF repo ships only
Q8_0 and f16 — no Q4_K_M — which is why Q8_0 was chosen for both.)

The prompt strings are loaded from disk once and the *same Python objects* are passed to both
models, so "byte-identical prompts across models" is true by construction, not by carefulness.
The notebook asserts it anyway. Same for the grammar between A2 and B2. `temperature = 0`,
`seed = 42` everywhere.

---

## Dataset

159 frames at 512×288, sampled at 1 fps from one clip, letterboxed to 512×512 (aspect preserved,
black bars — **never stretched**, because a distorted frame would corrupt every bbox coordinate).

The same person appears in three shirt colours. **Red is the target**; white and green are
**distractors** — the hardest realistic kind, where everything matches except the one attribute
in the target description. This distinction is what makes the false-positive metric meaningful,
and it cannot be derived from pixels, which is why `data/labeled_frames.html` is required input.

After dropping 10 culled frames: **149 frames — 29 target, 65 distractor, 55 empty.**

---

## Layout

```
Eval/
├── frames_square/                  your source frames (RAW_IMAGES_DIR)
└── vlm_eval/                       PROJECT_DIR
    ├── notebooks/vlm_2x2_eval.ipynb    the notebook you run
    ├── src/harness.py                  all the logic, 7 sections, read top to bottom
    ├── prompts/prompt_v1.txt           versioned experimental artifacts — edit without
    ├── prompts/prompt_v2.txt           touching code
    ├── grammars/bbox.gbnf
    ├── data/
    │   ├── labeled_frames.html         class labels in (shirt / usable / close-mid)
    │   ├── frames/                     512×512 letterboxed frames + frame_sources.json
    │   ├── labels.json                 tagging progress, saved after every frame
    │   └── manifest.json               one record per frame: class + GT bbox
    ├── results/
    │   ├── metrics.md / metrics.csv    the 4×6 table
    │   ├── error_analysis.md           worst FPs, missed targets, malformed dumps, verdict
    │   ├── run_info.json               provenance: date, seed, prompt SHA-1s
    │   └── raw/<cell>.jsonl            per-frame raw model output, parsed result, latency
    ├── logs/server_A.log, server_B.log
    ├── models/                         GGUF cache (on Drive, survives disconnects)
    └── bin/llama-server                cached binary (skips the rebuild)
```

---

## Running it (Colab)

1. Put the `vlm_eval` folder on Google Drive and your frames in `frames_square/`.
2. Open `notebooks/vlm_2x2_eval.ipynb`.
3. Run the Drive-mount cell.
4. Run `!pip install -q "opencv-contrib-python<5"` → **Runtime → Restart session** → re-run from
   the top. Colab's preinstalled OpenCV conflicts with the widgets; nothing is lost on restart
   because `labels.json` is already on Drive.
5. Check `PROJECT_DIR` and `RAW_IMAGES_DIR` in the CONFIG cell.
6. Run the rest top to bottom.

Roughly 30 minutes of model time for 149 frames × 4 cells, plus a one-off ~4 min llama.cpp build
and ~1 GB of model downloads.

**If Colab disconnects, just re-run from the top.** Frames already letterboxed are skipped,
boxes already confirmed are skipped, the binary is restored from `bin/`, models come from
`models/`, and each cell skips the frames already in its `.jsonl`.

To re-run the experiment without redoing intake, set `INTAKE_MODE = "skip"` and start at cell 0e.

### The single-file alternative

`run_eval.py` does the same experiment as one script — `python run_eval.py` — for running outside
Colab, in VS Code or locally. It derives ground-truth boxes automatically instead of using the
notebook's manual tagger. The notebook and the script are interchangeable; they write the same
`results/` files. Use whichever you prefer, but don't run both against the same folder at once.

---

## Configuration

Every knob is in the CONFIG cell. The ones that change what the experiment *means*:

| Constant | Purpose |
|---|---|
| `TARGET_DESCRIPTION` | Substituted into both prompts as `{target}`. Currently *"person wearing a red shirt"*. |
| `TARGET_SHIRT` | Which shirt label in the HTML counts as target. Every other person colour becomes a distractor. |
| `MALFORMED_POLICY` | How an unparseable v1 answer is scored. See below. |
| `SEED`, `MAX_TOKENS` | Reproducibility, and a bound on runaway prose in the v1 cells. |
| `MMPROJ_OVERRIDE` | Only needed if the smoke test reports the server can't take images. |

---

## The six metrics

1. **Match accuracy** — target-present/absent classified correctly.
2. **False-positive rate** — a *distractor* flagged as the target. **This is the metric that
   matters most**: it is the failure that makes the robot follow the wrong person.
3. **Mean bounding-box IoU** — over frames where both a prediction and a ground-truth box exist.
4. **Malformed-output rate** — responses failing schema validation. Expected to be **exactly 0**
   in A2/B2, because the grammar makes malformed output impossible to *generate* rather than
   something caught afterwards.
5. **Latency** — mean and p95 ms per inference.
6. **Peak memory** — RSS high-water mark of the model server process.

### Scoring rule for malformed answers

`MALFORMED_POLICY = "count_as_miss"` (default): an unparseable answer is scored **wrong** for that
frame; on a distractor frame it counts as a false positive. The reasoning is that on the robot, an
answer the parser can't read is a failed inference — there is no partial credit. The alternative,
`"exclude"`, drops those frames from the accuracy and FPR denominators so those two numbers
describe only well-formed answers. **The policy in force is printed at the top of `metrics.md`**,
so the accuracy figures are never ambiguous.

Either way the malformed *rate* is always computed over all answered frames, so nothing hides.

A request that fails outright (server down, HTTP error) is a harness problem, not a model answer:
excluded from every metric and counted separately as `n_request_errors`.

### The verdict rule

`error_analysis.md` ends with a recommendation, decided by comparing **A2 against B2** — the
grammar-constrained refined setup is what would actually run on the car. Lowest false-positive
rate wins; gaps under 2 pp count as a tie and fall through to accuracy, then IoU, then latency
(25 ms tolerance). It also reports the A1→A2 and B1→B2 deltas and checks the **interaction** —
whether refinement helped both models equally or mainly rescued the weaker one.

---

## Status of the current run (2026-09-17)

All four cells completed on Colab CPU. Wall-clock and peak memory, straight from the run log:

| Cell | Wall clock | Peak server RSS |
|---|---|---|
| A1 | 5.0 min | 7320 MB |
| A2 | 5.6 min | 7330 MB |
| B1 | 8.5 min | 3582 MB |
| B2 | 7.3 min | 3591 MB |

The authoritative accuracy / FPR / malformed numbers are in `results/metrics.md`.

### Two caveats you should know before quoting these numbers

**1. IoU is `n/a` for this run.** The tagger was skipped, so `manifest.json` has *"with GT bbox:
0"* — there are no ground-truth boxes to compare against, and the notebook warned about this
(*"93 frames not confirmed"*). The other five metrics are unaffected. To get an IoU column, either
work through cell 0c, or use `run_eval.py`, which derives boxes automatically and writes
`results/gt_boxes_check.jpg` so you can eyeball them. Be aware the automatic boxes are imperfect
on this clip — the camera's auto-exposure swings the mean brightness from 41 to 83 across the
frames, and a white shirt against a pale wall has very little contrast to detect.

**2. The peak-memory column is probably measuring llama.cpp's default context allocation, not the
models.** 7.3 GB for a 450M-parameter model at Q8_0 is implausible — the weights are around
500 MB. What is almost certainly being measured is the KV cache that llama-server allocates from
each model's *default* context length, which differs between the two models. As it stands the
comparison says more about defaults than about the models, and neither figure fits the Pi 5's
memory budget. **Before citing memory, re-run both servers with an explicit matching context
size** (add `--ctx-size 4096` to the `start_server` command for both) — that makes the column a
real apples-to-apples comparison and a number that means something for the robot.

Latency is Colab CPU, not a Pi 5. Only the *ratio* between the two models carries over.

---

## Design decisions, and why

Things you may be asked to justify:

- **Letterbox, never stretch.** The deployment model processes 512×512 natively. Squashing a 16:9
  frame into a square makes people ~1.8× too wide and puts every bbox in a distorted coordinate
  system. Padding preserves the geometry and matches exactly what the robot's camera path produces.
- **Same person, different shirt, as the distractor.** The realistic failure mode isn't confusing
  a person with a chair — it's confusing two people where only the described attribute differs.
  A distractor set of "other random people" would have made the task far too easy.
- **Identical prompts across models, identical grammar across the v2 cells.** Any difference would
  confound the model comparison — you would no longer know whether A beat B, or whether A's prompt
  beat B's prompt.
- **Both models at Q8_0 under one runtime.** Mixed quantization or two different servers would
  make the latency and memory columns meaningless.
- **Prompts and grammar live in files, not string literals.** The prompt is a versioned
  experimental artifact; you can diff it, swap it, and cite its SHA-1 (recorded in `run_info.json`).
- **v1 is deliberately weak.** No schema, no examples, no coordinate convention. That is the
  point: it establishes what you get without prompt engineering, so the v1→v2 delta measures
  something real.
- **Grammar-constrained decoding, not output repair.** The GBNF grammar restricts which tokens the
  model may emit, so malformed JSON is structurally impossible to produce — as opposed to
  generating freely and patching the damage afterwards, which leaves you guessing about intent.
- **Results written incrementally.** Every frame's answer hits disk as it arrives, so a
  disconnected Colab session costs minutes, not the whole run.
- **Raw model output is kept per frame.** `results/raw/<cell>.jsonl` stores the exact text
  alongside the parsed result, so any surprising number can be traced back to what the model
  literally said.

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `PROJECT_DIR not found` | Drive not mounted, or the path in CONFIG doesn't match where the folder actually is. |
| Widgets don't render in the tagger | The `opencv-contrib-python<5` install + **Runtime → Restart session** step was skipped. |
| Smoke test: HTTP 400 mentioning `grammar` | This llama.cpp build predates the `grammar` request field. Rebuild llama.cpp, or switch to `json_schema`. |
| Smoke test: error mentioning image / mmproj | The vision projector didn't load. Set `MMPROJ_OVERRIDE["A"]` or `["B"]` to the mmproj filename that `verify_hf_repo` printed, and restart that server. |
| A cell stops early with "5 requests failed in a row" | The server died — check `logs/server_*.log`. Partial results are kept; re-run the cell to resume. |
| `Model B unavailable` | Repo renamed or unreachable. The harness continues with A1/A2 only and says so rather than dying mid-experiment. |
| Metrics table shows `n/a` for IoU | No ground-truth boxes in the manifest. See caveat 1 above. |
| A cell reports 0 frames to go but you wanted a fresh run | Delete that cell's `results/raw/<cell>.jsonl` — resume is by frame id. |
