#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <export_root>" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPORT_ROOT="$1"
MODELS_DIR="$REPO_ROOT/models"

mkdir -p \
  "$MODELS_DIR/qnn" \
  "$MODELS_DIR/tokenizers/smolvlm_500m_instruct" \
  "$MODELS_DIR/tokenizers/whisper_tiny_en" \
  "$MODELS_DIR/voice"

copy_if_present() {
  local source_path="$1"
  local destination_path="$2"

  if [[ -e "$source_path" ]]; then
    mkdir -p "$(dirname "$destination_path")"
    cp -R "$source_path" "$destination_path"
    echo "Copied: $source_path -> $destination_path"
  else
    echo "Missing optional source: $source_path" >&2
  fi
}

copy_if_present \
  "$EXPORT_ROOT/vision_encoder_qnn.pte" \
  "$MODELS_DIR/qnn/vision_encoder_qnn.pte"

copy_if_present \
  "$EXPORT_ROOT/tok_embedding_qnn.pte" \
  "$MODELS_DIR/qnn/tok_embedding_qnn.pte"

copy_if_present \
  "$EXPORT_ROOT/hybrid_llama_qnn.pte" \
  "$MODELS_DIR/qnn/hybrid_llama_qnn.pte"

copy_if_present \
  "$EXPORT_ROOT/whisper_tiny_en.pte" \
  "$MODELS_DIR/qnn/whisper_tiny_en.pte"

copy_tokenizer_bundle() {
  local source_root="$1"
  local destination_root="$2"
  local copied=0
  local pattern

  mkdir -p "$destination_root"

  for pattern in \
    "tokenizer*" \
    "special_tokens_map.json" \
    "chat_template.jinja" \
    "preprocessor_config.json" \
    "processor_config.json" \
    "generation_config.json" \
    "added_tokens.json" \
    "*.model" \
    "vocab.json" \
    "merges.txt"; do
    local source_path
    for source_path in "$source_root"/$pattern; do
      if [[ -e "$source_path" ]]; then
        cp -R "$source_path" "$destination_root/"
        echo "Copied tokenizer asset: $source_path -> $destination_root/"
        copied=1
      fi
    done
  done

  if [[ "$copied" -eq 0 ]]; then
    echo "Missing tokenizer bundle in export root: $source_root" >&2
  fi
}

copy_tokenizer_bundle \
  "$EXPORT_ROOT" \
  "$MODELS_DIR/tokenizers/smolvlm_500m_instruct"

copy_if_present \
  "$EXPORT_ROOT/whisper_tiny_en" \
  "$MODELS_DIR/tokenizers/whisper_tiny_en"

copy_if_present \
  "$EXPORT_ROOT/en_US-lessac-medium.onnx" \
  "$MODELS_DIR/voice/en_US-lessac-medium.onnx"

copy_if_present \
  "$EXPORT_ROOT/en_US-lessac-medium.onnx.json" \
  "$MODELS_DIR/voice/en_US-lessac-medium.onnx.json"

echo ""
echo "Packaged artifacts under: $MODELS_DIR"
echo "Next step: python3 \"$REPO_ROOT/scripts/validate_runtime_config.py\""
