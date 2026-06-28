# SnapOn

**Point. Ask. Remember. All offline.**  
An on-device visual memory assistant for Samsung Galaxy S25 Ultra — built for hackathon June 27-28.

---

## Demo

```
[ screenshot placeholder — add docs/screenshot.png ]
```

---

## Problem Statement

We forget things constantly — where we parked, what the label says, who we met. Existing assistants
send your photos and voice to the cloud, where they are processed, logged, and stored.

SnapOn solves this with a **fully offline pipeline**:

- **Point** your camera at anything
- **Ask** a question by voice or text
- **Remember** facts that persist across sessions

No cloud. No account. No data leaving your device. Ever.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Vision + Text + Compress + Rerank + Embed | **InternVL3-1B** (HuggingFace Transformers, CPU bfloat16) |
| Speech-to-Text | faster-whisper `base.en` (CPU, INT8) |
| Text Embeddings | InternVL3-1B LLM backbone (mean-pool last hidden, L2-norm) |
| Vector Search | FAISS `IndexFlatIP` — cosine similarity (threshold 0.3) |
| Persistent Memory | SQLite3 |
| Text-to-Speech | piper-tts `en_US-lessac-medium` + sounddevice |
| Planned NPU Backend | Qualcomm QNN via ExecuTorch 1.3 (target: SM8750) |
| API Server | Flask + flask-cors + SSE streaming |
| UI | Vanilla HTML/CSS/JS — mobile-first, camera + waveform |

**One model does everything.** InternVL3-1B handles vision queries, text-only
queries, memory compression, reranking, AND memory embeddings — no second model,
no Ollama process, no network calls. The LLM backbone's hidden states serve as
the semantic vector space for FAISS retrieval, replacing the separate
all-MiniLM-L6-v2 (90 MB saved, one fewer model to maintain).

| Model | Job | Size |
|-------|-----|------|
| InternVL3-1B | Vision + text + compress + rerank + embed | ~2.0 GB (bfloat16) |
| faster-whisper base.en | Speech-to-text | ~145 MB |
| piper en_US-lessac-medium | Text-to-speech | ~61 MB |
| **Total** | | **~2.2 GB** |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        SnapOn Pipeline                        │
│                                                              │
│   🎤 Mic ──► faster-whisper ──► question text               │
│                                        │                     │
│   📷 Camera ──► image (JPEG) ──────────┤                     │
│                                        ▼                     │
│              SQLite + FAISS ──► InternVL3-1B ──► answer      │
│              (semantic memory)          │        (text)      │
│                    ▲                   ▼                     │
│                    │             PiperTTS ──► 🔊 Speaker     │
│              save_memory()                                   │
└──────────────────────────────────────────────────────────────┘

Ports
  :8000  Flask API  (server/app.py)
  :5000  UI proxy   (client/app.py)
```

---

## Setup

### Prerequisites

- Python 3.10+ (miniconda recommended)

```bash
pip install faster-whisper faiss-cpu flask flask-cors pillow \
            sounddevice numpy scipy piper-tts \
            transformers accelerate torch torchvision \
            sentence-transformers
```

- Piper TTS voice model (one-time download, ~61 MB):

```bash
mkdir -p ~/snapon/data/piper
BASE="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium"
wget -P ~/snapon/data/piper "$BASE/en_US-lessac-medium.onnx"
wget -P ~/snapon/data/piper "$BASE/en_US-lessac-medium.onnx.json"
```

> **InternVL3-1B** is downloaded automatically from HuggingFace on first startup
> (~2 GB, cached in `~/.cache/huggingface`). The pre-loader runs in a background
> thread so the server is responsive immediately; full model load takes ~60–90s
> on CPU. The on-device path is being built around the same model family, with
> local `.pte` artifacts tracked by `models/artifact_manifest.json`.

---

## Run

```bash
bash ~/snapon/scripts/start.sh
```

Open **http://localhost:5000** in your browser.

Or start manually:

```bash
# Terminal 1 — API server
cd ~/snapon
python -m server.app

# Terminal 2 — UI
python client/app.py
```

### Phone testing — wake word requires HTTPS

The **"Hey Snap" wake word** uses the Web Speech API (`SpeechRecognition`), which
Chrome and Edge only allow on **secure origins** (HTTPS or `localhost`).

When testing on a phone pointed at your dev machine over the local network you
will get a silent permission failure because `http://192.168.x.x:5000` is not
secure. Two options:

**Option A — ngrok tunnel (recommended)**

```bash
# Install once: https://ngrok.com/download
ngrok http 5000
# Open the https://xxxxx.ngrok-free.app URL on the phone
```

**Option B — Chrome flag (Android)**

1. On the phone open `chrome://flags/#unsafely-treat-insecure-origin-as-secure`
2. Add `http://<YOUR_WSL_IP>:5000` to the text box and enable the flag
3. Relaunch Chrome

Find your WSL IP with:
```bash
ip addr show eth0 | grep 'inet ' | awk '{print $2}' | cut -d/ -f1
```

### Test endpoints directly

```bash
# Health
curl http://localhost:8000/health

# Save a memory
curl -X POST http://localhost:8000/save \
  -H "Content-Type: application/json" \
  -d '{"text": "My name is Alekya, USC grad student", "tag": "personal"}'

# Query (text)
curl -X POST http://localhost:8000/query -F "question=who am I"

# Query (image + text)
curl -X POST http://localhost:8000/query \
  -F "question=What is in this photo?" \
  -F "image=@/path/to/photo.jpg"

# List memories
curl http://localhost:8000/memories

# Delete memory #1
curl -X DELETE http://localhost:8000/memories/1
```

---

## On-device deployment foundation (Samsung Galaxy S25 Ultra)

The checked-in app is still a desktop Python prototype. The on-device foundation
now lives in:

- `config/runtime_tasks.json` — task-by-task runtime matrix for QNN vs fallback
- `models/artifact_manifest.json` — expected local `.pte`, tokenizer, and voice assets
- `server/on_device_runtime.py` — callable runtime interface/stub for readiness and future bridge work
- `android/runtime/SnapOnRuntimeContract.kt` — Kotlin interface for the future Android runtime bridge
- `scripts/setup_executorch.sh` — ExecuTorch/QNN setup and export staging for SM8750

Phone inference is considered ready only when required artifacts are present and
validated by `python scripts/validate_runtime_config.py`.

### Model stack

| Model | Job | Asset | Runtime |
|---|---|---|---|
| SmolVLM-500M-Instruct | Vision encoder | `models/qnn/vision_encoder_qnn.pte` | QNN / ExecuTorch |
| SmolVLM-500M-Instruct | Token embedding | `models/qnn/tok_embedding_qnn.pte` | QNN / ExecuTorch |
| SmolVLM-500M-Instruct | Text decoder | `models/qnn/hybrid_llama_qnn.pte` | QNN / ExecuTorch |
| Whisper tiny.en | Speech to text | `models/qnn/whisper_tiny_en.pte` | QNN / ExecuTorch, optional for MVP |
| PiperTTS | Text to speech | `models/voice/en_US-lessac-medium.onnx` | Local CPU, optional |

### Why SmolVLM-500M-Instruct

SmolVLM-500M-Instruct is the active Qualcomm/ExecuTorch path for the on-device
VLM demo because Qualcomm already wires it through the `llama.py` multimodal
export flow. Personal memory stays as a lightweight local layer on top until a
separate Qualcomm-friendly embedding model is selected.

### Export commands (run on-site with QNN SDK)

ExecuTorch repo: `github.com/pytorch/executorch`

```bash
# Set SDK path
export QNN_SDK_ROOT=/path/to/qairt/2.x.x.xxxxxxx

DEVICE_SERIAL=<adb_serial> bash scripts/setup_executorch.sh
```

Device target: SM8750 (Galaxy S25 Ultra — Snapdragon 8 Elite)

---

## Judging Criteria Alignment

| Criterion | SnapOn |
|-----------|--------|
| **NPU Utilization** | Runtime matrix and artifact manifest define the QNN path for VLM, embeddings, and optional Whisper; current checked-in server remains a desktop fallback until artifacts/Android bridge land |
| **Privacy** | Fully air-gapped — no network calls, no telemetry, all data stored locally in SQLite |
| **Innovation** | Persistent cross-session visual memory with semantic retrieval (RAG on mobile) — ask about things you saw days ago |
| **Performance** | Planned INT4/INT8 QNN artifacts; FAISS cosine search remains the desktop prototype retrieval path |
| **User Experience** | Voice in → spoken answer out; zero setup for end user |

---

## Team

<!-- Add team members here -->

---

## License

MIT
