# Person-Following Robot — Vision Dataset (v1)

IT 244 / IT 344 — Task 1
Illinois State University

---

## Project

An autonomous robot car that searches for, follows, and hides from a specific
person described in a text field. No manual driving; the text description is the
only human input.

**Platform:** Hiwonder MentorPi M1 — Raspberry Pi 5, ROS2 Humble, mecanum wheels,
TOF lidar, depth camera.

**Models:** LFM2.5-VL-450M (quantized GGUF, llama.cpp) verifies whether a sampled
camera frame contains the described person. LFM2.5-230M acts as mission planner,
issuing prioritized goals to a ROS2 action-server layer.

This repository covers the vision side: the data used to measure whether the VLM's
verification is reliable enough to act on.

---

## What this dataset is for

This is an **evaluation set, not a training set.** Nothing here is used to fine-tune
a model. The purpose is to measure how well the off-the-shelf VLM answers the one
question the robot depends on: *is the person in this frame the person I was told
to find?*

That framing is why the set is small. It needs enough variation to expose failure
modes, not enough volume to learn from.

---

## Source

| | |
|---|---|
| Source | Self-recorded webcam video (`WIN_20260914_18_07_56_Pro.mp4`) |
| Resolution | 1920×1080 |
| Frame rate | 16.5 fps |
| Duration | 2 min 39 s (2,717 frames) |
| Subjects | One person, filmed in three different shirts |
| Location | Single bedroom, fixed camera position |

Recorded by the project author, with consent from the person filmed. No third-party
or scraped data. Raw video is excluded from the repository (see `.gitignore`); only
the processed frames are committed.

---

## Cleaning

1. **Sampled at 1 fps** — 2,717 source frames reduced to 159. Consecutive video
   frames are near-duplicates and add no information.
2. **Resized to 512 px on the long side, aspect ratio preserved** (1920×1080 →
   512×288). No padding, no stretching.
3. **Manual review** of all 159 frames; 10 motion-blurred or transitional frames
   flagged.
4. **Hand-labeled** into `labels.csv`.

**Final size:** 159 frames, ~2.9 MB. 149 marked usable.

### Why 512, and why not square

The VLM's vision encoder (SigLIP2 NaFlex) processes images up to 512×512 without
upscaling, and splits anything larger into multiple 512×512 tiles plus a thumbnail.
Staying at or under 512 on the long side keeps each frame to a single tile —
the minimum token count and the fastest inference, which matters for a robot making
decisions in a control loop.

The encoder handles non-square aspect ratios natively, so padding to 512×512 would
add ~112 px of empty bars top and bottom. Those bars still get encoded into tokens.
Preserving the native 16:9 ratio gives the same real pixels at a lower cost.

**The governing rule:** preprocessing here must match preprocessing at runtime. If
the evaluation frames are shaped differently from what the robot's depth camera
produces, accuracy measured here predicts nothing about the deployed system. The
final resize target will be set from the depth camera's output format.

### Culled frames

Blurred frames are marked `usable=0` rather than deleted, so the cleaning decision
stays visible and reversible.

---

## Labels

`labels.csv`, one row per frame:

| Column | Values |
|---|---|
| `filename` | `frame_0001.jpg` … `frame_0159.jpg` |
| `shirt_color` | `white`, `red`, `green`, `gray`, `none` |
| `person_present` | `1` / `0` |
| `distance` | `close`, `mid`, `n/a` |
| `room` | `bedroom_01` |
| `lighting` | `daylight_window` |
| `clip_id` | `clip01` |
| `usable` | `1` / `0` |
| `notes` | free text |

### Distribution

| Class | All frames | Usable |
|---|---|---|
| No person | 55 | 55 |
| White shirt | 43 | 40 |
| Red shirt | 32 | 29 |
| Green shirt | 28 | 25 |
| Gray shirt | 1 | 0 |
| **Total** | **159** | **149** |

### What the labels actually mean

The clip contains **one person in three shirts** — not a target and a distractor.
So `shirt_color` is a label about clothing, not about identity.

This is named honestly rather than relabeled as `target` / `distractor`, because
calling it that would claim the dataset tests person identification when it tests
color discrimination. A model can score perfectly here by finding the color white
and never looking at a face.

That still makes it a usable v1: it establishes the pipeline, the label schema, and
a measurable baseline. v2 adds a second person in a similar-colored shirt, which is
what separates *identity* from *clothing*.

---

## Known limitations

- **No distractor person.** The single largest gap. Addressed in v2.
- **One environment.** Same room, same wall, same fixed camera for all 159 frames.
  A model could learn the background instead of the subject.
- **Static camera at standing height.** The robot's camera is low and moving. No
  frame here matches the viewpoint the robot will actually have.
- **Correlated frames.** All frames come from one continuous clip. Per-frame
  accuracy will overstate real performance; results should be reported per-clip as
  well, and any train/test split must divide by clip, never by frame.
- **Empty class is misleading.** 55 frames of an identical static shot. Numerically
  the largest class; informationally close to a single frame.
- **Uncontrolled variable.** The subject changes from pants to shorts partway
  through the white-shirt segment.

---

## Success criteria

Overall accuracy is the wrong headline metric. A robot that confidently follows the
wrong person is worse than one that follows nobody, so the metric that matters is
the **false-positive rate on non-target people**.

The VLM must also return a usable *uncertainty* signal. When confidence is low, the
planner should hold a "keep searching" state rather than committing to a follow.

---

## Repository layout

```
data/
  frames/          159 JPEGs, 512×288
  labels.csv       one row per frame
  search_logs.jsonl  authored corpus (RAG component, later task)
docs/
  workflow.png     envisioned system workflow
README.md
```

---

## Next steps

1. Reshoot with a second person in a similar-colored shirt.
2. Add two or three rooms and vary lighting.
3. Re-record at robot camera height, with the camera moving.
4. Confirm the depth camera's output format and fix the resize target to match.
5. Baseline the VLM against v1 to establish the measurement harness.
