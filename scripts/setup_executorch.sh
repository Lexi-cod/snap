#!/usr/bin/env bash
# Bootstrap the ExecuTorch + Qualcomm QNN export workspace for SnapOn.
#
# This script prepares tooling and stages export directories for the planned
# Galaxy S25 Ultra target. It does not claim success unless the QNN SDK is
# installed and the upstream export entrypoints exist in the checked-out
# ExecuTorch release.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXECUTORCH_BRANCH="${EXECUTORCH_BRANCH:-release/1.3}"
EXECUTORCH_DIR="${EXECUTORCH_DIR:-$HOME/executorch}"
VENV_DIR="${VENV_DIR:-$HOME/executorch-env}"
TARGET_SOC="${TARGET_SOC:-SM8750}"
DEVICE_SERIAL="${DEVICE_SERIAL:-}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-$REPO_ROOT/models}"
QNN_ARTIFACT_DIR="$ARTIFACT_ROOT/qnn"
TOKENIZER_DIR="$ARTIFACT_ROOT/tokenizers"

mkdir -p "$QNN_ARTIFACT_DIR" "$TOKENIZER_DIR" "$ARTIFACT_ROOT/voice"

if [[ ! -d "$VENV_DIR" ]]; then
  python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

if [[ ! -d "$EXECUTORCH_DIR/.git" ]]; then
  git clone -b "$EXECUTORCH_BRANCH" https://github.com/pytorch/executorch "$EXECUTORCH_DIR"
fi

cd "$EXECUTORCH_DIR"
git submodule sync
git submodule update --init --recursive
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -r requirements-examples.txt
python -m pip install -e .

if [[ -z "${QNN_SDK_ROOT:-}" ]]; then
  cat <<MSG

ExecuTorch tooling is installed, but QNN export/build was not run.

Set QNN_SDK_ROOT to the Qualcomm AI Engine Direct SDK path, then rerun:

  QNN_SDK_ROOT=/path/to/qairt DEVICE_SERIAL=<adb_serial> $0

MSG
  exit 0
fi

if [[ -z "$DEVICE_SERIAL" ]]; then
  echo "DEVICE_SERIAL is required for Qualcomm OSS export scripts." >&2
  exit 2
fi

./backends/qualcomm/scripts/build.sh

run_if_present() {
  local script_path="$1"
  shift
  if [[ -f "$script_path" ]]; then
    python "$script_path" "$@"
  else
    echo "Skipping missing upstream export script: $script_path" >&2
  fi
}

# Planned primary path: Qualcomm SmolVLM for visual Q&A and scene description.
run_if_present \
  examples/qualcomm/oss_scripts/llama/llama.py \
  -b build-android \
  -s "$DEVICE_SERIAL" \
  -m "$TARGET_SOC" \
  --decoder_model smolvlm_500m_instruct \
  --model_mode hybrid \
  --prefill_ar_len 16 \
  --max_seq_len 1024 \
  --prompt "Can you describe this image?" \
  --artifact "$QNN_ARTIFACT_DIR/smolvlm_500m_instruct" \
  --compile_only

# Planned STT path. Text input remains the fallback until this artifact is
# exported and measured on-device.
run_if_present \
  examples/qualcomm/oss_scripts/whisper/whisper.py \
  -b build-android \
  -s "$DEVICE_SERIAL" \
  -m "$TARGET_SOC" \
  --artifact "$QNN_ARTIFACT_DIR/whisper_tiny_en"

cat <<MSG

ExecuTorch/QNN setup finished.

Expected SnapOn asset locations:
  $QNN_ARTIFACT_DIR/smolvlm_500m_instruct/hybrid_llama_qnn.pte
  $QNN_ARTIFACT_DIR/smolvlm_500m_instruct/vision_encoder_qnn.pte
  $QNN_ARTIFACT_DIR/smolvlm_500m_instruct/tok_embedding_qnn.pte
  $QNN_ARTIFACT_DIR/whisper_tiny_en.pte
  $QNN_ARTIFACT_DIR/smolvlm_500m_instruct/tokenizer*.json
  $TOKENIZER_DIR/whisper_tiny_en/
  $ARTIFACT_ROOT/voice/en_US-lessac-medium.onnx
  $ARTIFACT_ROOT/voice/en_US-lessac-medium.onnx.json

After copying artifacts, update models/artifact_manifest.json with real sha256
values and run:

  python scripts/validate_runtime_config.py

MSG
