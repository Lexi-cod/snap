# SnapOn On-Device Implementation Gaps

This repo is not yet a true Samsung Galaxy S25 / Snapdragon / ExecuTorch app.
Right now it is a desktop-hosted Python + Flask prototype with a browser UI.

## Current state

- UI runs in a browser via `client/app.py` and `templates/index.html`.
- Inference runs in Python on the host machine via `server/app.py` and `server/pipeline.py`.
- Persistence uses local SQLite + FAISS files in `~/snapon/data`.
- The README claims QNN / ExecuTorch / on-device deployment, but the codebase does not yet implement that runtime.

## Blocking gaps for hackathon compliance

### 1. No Android app shell

What exists:
- Browser UI only: `client/app.py`
- Web page only: `templates/index.html`

What is missing:
- Android Studio project
- Kotlin/Java app entrypoint
- CameraX integration
- microphone capture on Android
- local asset packaging for models and indexes
- on-device storage wiring for memories

Why it matters:
- The current demo depends on a laptop web server and localhost proxy, not a phone-native app.

### 2. No ExecuTorch runtime in the product path

What exists:
- Python `transformers` load path in `server/pipeline.py`
- README export notes
- `scripts/setup_executorch.sh` only installs tooling

What is missing:
- checked-in `.pte` model artifacts or export pipeline outputs
- Android ExecuTorch integration
- Qualcomm QNN backend initialization in app runtime
- model session loading and inference calls from the phone app
- fallback/error handling for QNN load failures

Why it matters:
- The implementation currently runs HuggingFace PyTorch models in Python, not ExecuTorch on Snapdragon.

### 3. Core inference still runs on desktop/server Python

Current implementation:
- VLM: `AutoModel.from_pretrained(...)` in `server/pipeline.py`
- STT: `WhisperModel("base.en", device="cpu", compute_type="int8")`
- visual retrieval: `open_clip`
- HTTP API: Flask in `server/app.py`

What is missing:
- all inference code moved into an on-device runtime layer
- removal of Flask as a required serving boundary
- local direct function calls from app UI to inference engine

Why it matters:
- "Majority of workload must be on-device" is not satisfied by a phone browser talking to a Python server.

### 4. Wake word and TTS depend on browser features, not the target stack

Current implementation:
- wake word uses `SpeechRecognition` / `webkitSpeechRecognition` in `templates/index.html`
- answer speech uses browser `speechSynthesis`

What is missing:
- offline wake word or push-to-talk flow inside the Android app
- offline TTS integrated in app runtime
- Android audio focus / playback / recording control

Why it matters:
- `SpeechRecognition` is browser-dependent and often not truly offline.
- Browser TTS is outside the intended PyTorch + ExecuTorch + Snapdragon story.

### 5. Model stack does not match the actual deployment story yet

Current implementation:
- README says InternVL3-1B, Whisper, FAISS, Piper, QNN
- code actually uses InternVL3, faster-whisper, CLIP, Piper, FAISS in Python

What is missing:
- a verified set of models that all export and run on the S25 target
- a single documented inference matrix:
  - model name
  - task
  - export script
  - quantization
  - runtime backend
  - measured latency on device

Why it matters:
- Judges will ask what is really running on Hexagon today versus what is aspirational.

### 6. No packaged offline-first bootstrap

Current implementation:
- first-run model downloads from HuggingFace
- Piper voice download instructions in README

What is missing:
- bundled model assets or a one-time offline sideload flow
- app startup that works without internet
- local validation that all required assets are present before demo

Why it matters:
- A hackathon demo can fail if the phone needs network to fetch model weights.

## Important consistency gaps

### README overclaims implementation status

The README currently says:
- all models compile to ExecuTorch `.pte`
- runs entirely on Hexagon NPU
- all inference runs on Snapdragon 8 Elite NPU

The checked-in code does not implement those claims yet.

### Setup script points at the wrong export path

`scripts/setup_executorch.sh` ends with a `llama.py` / `qwen2_5_vl_3b` export example, while the repo narrative is built around InternVL3.

This needs one coherent export path for the actual model you will demo.

## Recommended implementation shape for the hackathon

### Phase 1: Make it a true phone demo

Build these first:
- Android app shell in Kotlin
- CameraX preview + frame capture
- push-to-talk button
- local SQLite memory store on device
- direct on-device inference bridge

Do not keep for the core demo:
- Flask server
- browser proxy
- localhost split architecture

### Phase 2: Tighten the model scope

Best realistic demo scope:
- one visual understanding model on ExecuTorch/QNN
- one speech model on ExecuTorch/QNN or a clearly minor CPU fallback
- SQLite memory persistence
- simple retrieval layer

Avoid for the first judging demo:
- too many separate models
- browser wake-word logic
- desktop-only Python dependencies

### Phase 3: Replace the current runtime pieces

Replace:
- `templates/index.html` wake word with Android push-to-talk
- browser `speechSynthesis` with packaged offline TTS
- Flask endpoints with in-app use cases
- HuggingFace runtime loading with local ExecuTorch model loading

Keep conceptually:
- memory save / query flow
- retrieval before answer generation
- image-linked memories

## Concrete file-by-file implications

### `client/app.py`
- Remove from the final phone architecture.
- It is only a localhost proxy.

### `templates/index.html`
- Use as a UX reference only.
- Port the flow to Android native UI.

### `server/app.py`
- Good for prototyping the product behavior.
- Not suitable as the final hackathon runtime architecture.

### `server/pipeline.py`
- Useful as behavior spec for the inference pipeline.
- Needs to be split into:
  - model runner interface
  - Android/ExecuTorch-backed implementation
  - optional desktop dev stub

### `server/memory.py`
- The SQLite memory logic is reusable.
- FAISS may remain if you can package it on Android, otherwise replace with a lighter retrieval approach.

### `scripts/setup_executorch.sh`
- Needs to become a real export/build pipeline for the exact chosen model set.

## Fastest path to "truly on-device"

1. Create an Android app.
2. Move camera, mic, and TTS into Android-native code.
3. Pick one supported QNN-friendly model stack and export it.
4. Load `.pte` models locally from app storage/assets.
5. Replace HTTP calls with direct app-layer inference calls.
6. Keep memory storage entirely local.
7. Measure and record on-device latency for the demo and slides.

## Suggested MVP for this repo

If time is tight, the most realistic implementation story is:
- on-device image understanding
- on-device speech-to-text
- local memory save/retrieve
- short spoken answer

Skip for MVP unless already working:
- wake word
- multi-model orchestration beyond what the device can reliably demo
- browser-only features

## Bottom line

The idea fits the hackathon well, but the current codebase is still a desktop prototype.
The main missing component is not a small patch: it is the actual Android + ExecuTorch runtime layer.
