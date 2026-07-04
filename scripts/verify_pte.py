#!/usr/bin/env python3
"""
Verify that the exported SmolVLM XNNPACK .pte artifacts load and produce
correct output shapes via the ExecuTorch Python runtime.

Usage:
  python scripts/verify_pte.py [--pte-dir models/xnnpack]
"""

import argparse
import sys
from pathlib import Path

import torch
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

IMAGE_SIZE = 512
N_VIS_TOKENS = 64
HIDDEN_DIM = 960
VOCAB_SIZE = 49280
# Must match the --max-seq-len used when the .pte artifacts were exported.
# Current artifacts: max_seq_len=128 (64 visual tokens + up to 64 text tokens).
SAMPLE_SEQ = 128


def load_pte(path: Path):
    from executorch.runtime import Runtime, Verification

    runtime = Runtime.get()
    program = runtime.load_program(str(path), verification=Verification.Minimal)
    return program.load_method("forward")


def check_vision_encoder(pte_dir: Path) -> bool:
    path = pte_dir / "vision_encoder.pte"
    print(f"\n[vision_encoder] loading {path.name} …")
    try:
        method = load_pte(path)
        dummy = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.float32)
        out = method.execute([dummy])
        feat = out[0]
        expected = (1, N_VIS_TOKENS, HIDDEN_DIM)
        ok = tuple(feat.shape) == expected
        status = "PASS" if ok else f"FAIL (got {tuple(feat.shape)}, expected {expected})"
        print(f"  output shape: {tuple(feat.shape)}  → {status}")
        return ok
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def check_tok_embedding(pte_dir: Path) -> bool:
    path = pte_dir / "tok_embedding.pte"
    print(f"\n[tok_embedding] loading {path.name} …")
    try:
        method = load_pte(path)
        dummy_ids = torch.zeros(1, SAMPLE_SEQ, dtype=torch.int64)
        out = method.execute([dummy_ids])
        emb = out[0]
        expected = (1, SAMPLE_SEQ, HIDDEN_DIM)
        ok = tuple(emb.shape) == expected
        status = "PASS" if ok else f"FAIL (got {tuple(emb.shape)}, expected {expected})"
        print(f"  output shape: {tuple(emb.shape)}  → {status}")
        return ok
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def check_decoder(pte_dir: Path) -> bool:
    path = pte_dir / "smolvlm_decoder.pte"
    print(f"\n[smolvlm_decoder] loading {path.name} …")
    try:
        method = load_pte(path)
        dummy_embeds = torch.zeros(1, SAMPLE_SEQ, HIDDEN_DIM, dtype=torch.float32)
        dummy_pos = torch.arange(SAMPLE_SEQ, dtype=torch.int64).unsqueeze(0)
        out = method.execute([dummy_embeds, dummy_pos])
        logits = out[0]
        expected = (1, SAMPLE_SEQ, VOCAB_SIZE)
        ok = tuple(logits.shape) == expected
        status = "PASS" if ok else f"FAIL (got {tuple(logits.shape)}, expected {expected})"
        print(f"  output shape: {tuple(logits.shape)}  → {status}")
        return ok
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pte-dir", default=str(REPO_ROOT / "models" / "xnnpack"))
    args = p.parse_args()
    pte_dir = Path(args.pte_dir)

    if not pte_dir.exists():
        print(f"ERROR: {pte_dir} not found — run scripts/export_smolvlm_pte.py first")
        sys.exit(1)

    results = {
        "vision_encoder": check_vision_encoder(pte_dir),
        "tok_embedding":  check_tok_embedding(pte_dir),
        "smolvlm_decoder": check_decoder(pte_dir),
    }

    print("\n── Summary ──────────────────────────────")
    all_ok = True
    for name, ok in results.items():
        icon = "✓" if ok else "✗"
        print(f"  {icon}  {name}")
        all_ok = all_ok and ok

    if all_ok:
        print("\nAll artifacts verified — ready for Android packaging.\n")
        sys.exit(0)
    else:
        print("\nSome artifacts failed. Re-run export_smolvlm_pte.py.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
