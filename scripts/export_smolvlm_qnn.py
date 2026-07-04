#!/usr/bin/env python3
"""
Export SmolVLM-500M-Instruct to ExecuTorch QNN .pte for Hexagon NPU (SM8750).

This script re-uses the same model decomposition as export_smolvlm_pte.py
(3 separate modules: vision encoder, tok embedding, decoder) but delegates
ops to the Qualcomm HTP via QnnPartitioner instead of XnnpackPartitioner.

Output (models/qnn/):
  vision_encoder_qnn.pte
  tok_embedding_qnn.pte
  smolvlm_decoder_qnn.pte

Prerequisites:
  • ExecuTorch built with EXECUTORCH_BUILD_QNN=ON (done by setup_qnn.sh)
  • QNN_SDK_ROOT set to QAIRT 2.46.0 path
  • build-android/ directory from ./backends/qualcomm/scripts/build.sh

Usage (called from setup_qnn.sh):
  python scripts/export_smolvlm_qnn.py \
      --build-path ~/executorch/build-android \
      --qnn-sdk-root /mnt/c/Users/aleky/OneDrive/Desktop/qairt/2.26.2.240911 \
      --soc-model SM8750 \
      --out-dir models/qnn \
      --max-seq-len 128
"""

import argparse
import hashlib
import json
import os
import sys
import warnings
from pathlib import Path

import torch
import torch.nn as nn

warnings.filterwarnings("ignore")

REPO_ROOT    = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO_ROOT / "models" / "artifact_manifest.json"
MODEL_ID     = "HuggingFaceTB/SmolVLM-500M-Instruct"
HIDDEN_DIM   = 960


# ── Reuse model classes from the XNNPACK export ──────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from export_smolvlm_pte import (
    FixedVisionEmbeddings,
    VisionEncoderModule,
    TokEmbeddingModule,
    DecoderPrefillModule,
    load_model,
)


# ── QNN export helper ─────────────────────────────────────────────────────────

def _export_to_qnn_pte(module, example_args, build_path: str,
                        qnn_sdk_root: str, soc_model: str,
                        label: str = "") -> bytes:
    """
    Export a module using QnnPartitioner for the target SoC.

    Key differences vs XNNPACK:
      • QnnPartitioner targets Hexagon HTP (AI accelerator on Snapdragon)
      • SM8750 (Galaxy S25) uses HTP v79 backend
      • Ops that don't map to HTP fall back to CPU automatically
    """
    from torch.export import export
    from executorch.exir import to_edge_transform_and_lower, EdgeCompileConfig
    from executorch.backends.qualcomm.partitioner.qualcomm_partitioner import (
        QualcommPartitioner,
    )
    from executorch.backends.qualcomm.utils.utils import (
        generate_htp_compiler_spec,
        get_htp_compile_spec,
    )

    print(f"    Exporting with QNN (HTP) for {soc_model} …")

    # HTP compiler spec — fp16 precision for best NPU throughput on SM8750
    backend_options = generate_htp_compiler_spec(use_fp16=True)
    compile_spec    = get_htp_compile_spec(soc_model, backend_options)
    partitioner     = QualcommPartitioner(compile_spec)

    exported = export(module, example_args, strict=True)

    edge = to_edge_transform_and_lower(
        exported,
        partitioner=[partitioner],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )
    et_program = edge.to_executorch()
    return et_program.buffer


def _write_pte(buf: bytes, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buf)
    sha = hashlib.sha256(buf).hexdigest()
    size_mb = len(buf) / 1024 / 1024
    print(f"    wrote {path.name}  ({size_mb:.1f} MB)  sha256={sha[:16]}…")
    return sha


# ── Per-module export functions ───────────────────────────────────────────────

def export_vision_encoder_qnn(model, out_dir: Path, build_path: str,
                               qnn_sdk_root: str, soc_model: str) -> str:
    print("\n=== Vision Encoder (QNN) ===")
    mod = VisionEncoderModule(model).eval()
    dummy = (torch.zeros(1, 3, 512, 512),)
    buf = _export_to_qnn_pte(mod, dummy, build_path, qnn_sdk_root, soc_model,
                               label="vision_encoder_qnn")
    return _write_pte(buf, out_dir / "vision_encoder_qnn.pte")


def export_tok_embedding_qnn(model, out_dir: Path, max_seq_len: int,
                              build_path: str, qnn_sdk_root: str,
                              soc_model: str) -> str:
    print(f"\n=== Token Embedding (QNN, seq={max_seq_len}) ===")
    mod = TokEmbeddingModule(model).eval()
    dummy = (torch.zeros(1, max_seq_len, dtype=torch.int64),)
    buf = _export_to_qnn_pte(mod, dummy, build_path, qnn_sdk_root, soc_model,
                               label="tok_embedding_qnn")
    return _write_pte(buf, out_dir / "tok_embedding_qnn.pte")


def export_decoder_qnn(model, out_dir: Path, max_seq_len: int,
                        build_path: str, qnn_sdk_root: str,
                        soc_model: str) -> str:
    print(f"\n=== Decoder (QNN, seq={max_seq_len}) ===")
    mod = DecoderPrefillModule(model).eval()
    dummy = (
        torch.zeros(1, max_seq_len, HIDDEN_DIM),
        torch.arange(max_seq_len, dtype=torch.int64).unsqueeze(0),
    )
    buf = _export_to_qnn_pte(mod, dummy, build_path, qnn_sdk_root, soc_model,
                               label="smolvlm_decoder_qnn")
    return _write_pte(buf, out_dir / "smolvlm_decoder_qnn.pte")


# ── Manifest update ───────────────────────────────────────────────────────────

def update_manifest(sha_vis: str, sha_tok: str, sha_dec: str,
                    out_dir: Path, soc_model: str):
    manifest = json.loads(MANIFEST_PATH.read_text())
    entries = {a["id"]: a for a in manifest["artifacts"]}

    new_entries = [
        {
            "id":       "smolvlm_vision_encoder_qnn",
            "task":     "visual_question_answering",
            "runtime":  "executorch_qnn_htp",
            "path":     "models/qnn/vision_encoder_qnn.pte",
            "required": True,
            "sha256":   sha_vis,
            "source":   MODEL_ID,
            "notes":    f"SigLIP ViT + Idefics3Connector. QNN HTP {soc_model}. "
                        "Input: [1,3,512,512] float16. Output: [1,64,960] float16.",
        },
        {
            "id":       "smolvlm_tok_embedding_qnn",
            "task":     "visual_question_answering",
            "runtime":  "executorch_qnn_htp",
            "path":     "models/qnn/tok_embedding_qnn.pte",
            "required": True,
            "sha256":   sha_tok,
            "source":   MODEL_ID,
            "notes":    "Token embedding table. QNN HTP. Input: [1,128] int64. Output: [1,128,960].",
        },
        {
            "id":       "smolvlm_decoder_qnn",
            "task":     "visual_question_answering",
            "runtime":  "executorch_qnn_htp",
            "path":     "models/qnn/smolvlm_decoder_qnn.pte",
            "required": True,
            "sha256":   sha_dec,
            "source":   MODEL_ID,
            "notes":    "SmolLM2 decoder prefill, no KV cache. QNN HTP. "
                        "Input: embeds [1,128,960] + pos [1,128]. Output: logits [1,128,49280].",
        },
    ]

    for entry in new_entries:
        entries[entry["id"]] = entry

    manifest["artifacts"] = list(entries.values())
    manifest["qnn_generated"] = True
    manifest["qnn_soc_model"] = soc_model
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    print(f"\n  Manifest updated: {MANIFEST_PATH}")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--build-path",    required=True,
                   help="Path to ExecuTorch build-android/ directory.")
    p.add_argument("--qnn-sdk-root",  required=True,
                   help="Path to QAIRT SDK root (e.g. ~/qairt/2.46.0.260424).")
    p.add_argument("--soc-model",     default="SM8750",
                   help="Qualcomm SoC target (default: SM8750 = Galaxy S25).")
    p.add_argument("--out-dir",       default="models/qnn")
    p.add_argument("--max-seq-len",   type=int, default=128)
    p.add_argument("--cache-dir",     default=None)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    build_path = args.build_path
    qnn_sdk = args.qnn_sdk_root

    print(f"SoC target   : {args.soc_model}")
    print(f"Max seq len  : {args.max_seq_len}")
    print(f"Build path   : {build_path}")
    print(f"QNN SDK root : {qnn_sdk}")
    print(f"Output dir   : {out_dir}")

    # Load model in fp32; QNN partitioner handles fp16 cast internally
    model = load_model(args.cache_dir, quantize=False)

    with torch.no_grad():
        sha_vis = export_vision_encoder_qnn(
            model, out_dir, build_path, qnn_sdk, args.soc_model)

        # Free vision weights before heavy decoder export
        del model.model.vision_model
        del model.model.connector
        import gc; gc.collect()

        sha_tok = export_tok_embedding_qnn(
            model, out_dir, args.max_seq_len, build_path, qnn_sdk, args.soc_model)

        sha_dec = export_decoder_qnn(
            model, out_dir, args.max_seq_len, build_path, qnn_sdk, args.soc_model)

    update_manifest(sha_vis, sha_tok, sha_dec, out_dir, args.soc_model)

    print("\n✓ QNN export complete.")
    print(f"  {out_dir}/vision_encoder_qnn.pte")
    print(f"  {out_dir}/tok_embedding_qnn.pte")
    print(f"  {out_dir}/smolvlm_decoder_qnn.pte")


if __name__ == "__main__":
    main()
