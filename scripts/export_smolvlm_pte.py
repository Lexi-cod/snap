#!/usr/bin/env python3
"""
Export SmolVLM-500M-Instruct to ExecuTorch XNNPACK .pte artifacts.

Architecture (from config):
  Vision:  SigLIP ViT — image_size=512, patch_size=16 → 32×32=1024 patches
           Idefics3Connector scale_factor=4 → 64 visual tokens @ dim 960
  Decoder: 32 layers, dim 960, 15 heads, 5 KV heads (GQA), vocab 49280

Outputs (models/xnnpack/):
  vision_encoder.pte   — SigLIP ViT + Idefics3Connector
  tok_embedding.pte    — token embedding table lookup
  smolvlm_decoder.pte  — SmolLM2 decoder, prefill mode (use_cache=False)

Backend: XNNPACK (ARM-optimised CPU kernels, no QNN SDK required).
For Hexagon NPU deployment run scripts/setup_executorch.sh with QNN_SDK_ROOT.

Usage:
  python scripts/export_smolvlm_pte.py [--max-seq-len 1024] [--no-quantize]
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

# ── ExecuTorch 1.3.x compat patch ────────────────────────────────────────────
# spec_prop_pass calls dim_order_from_stride() which tries to sort() strides
# using guard_size_oblivious comparisons.  After XNNPACK lowering, delegate
# output FakeTensors can have symbolic strides; sorted() receives SymBool
# values it cannot use, crashing with "TypeError: '<' not supported between
# instances".  We replace dim_order_from_stride with a version that catches the
# comparison failure and falls back to row-major (C-contiguous) dim order.
def _patched_dim_order_from_stride(stride):
    try:
        import typing
        from torch.utils._sympy.value_ranges import bound_sympy
        from torch.fx.experimental.symbolic_shapes import guard_size_oblivious

        class K(typing.NamedTuple):
            stride: int
            def __lt__(self, other):
                return guard_size_oblivious(self.stride < other.stride)
            def __gt__(self, other):
                return guard_size_oblivious(self.stride > other.stride)
            def __le__(self, other):
                return guard_size_oblivious(self.stride <= other.stride)
            def __ge__(self, other):
                return guard_size_oblivious(self.stride >= other.stride)
            def __eq__(self, other):
                return guard_size_oblivious(self.stride == other.stride)

        sorted_dims = [
            i[0] for i in sorted(enumerate(stride), key=lambda x: K(x[1]), reverse=True)
        ]
        return tuple(typing.cast(typing.Tuple[bytes], sorted_dims))
    except Exception:
        # Fallback: row-major C-contiguous order (0, 1, 2, …)
        return tuple(range(len(stride)))


import executorch.exir.tensor as _et_tensor
_et_tensor.dim_order_from_stride = _patched_dim_order_from_stride
# ─────────────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_ID = "HuggingFaceTB/SmolVLM-500M-Instruct"
OUT_DIR = REPO_ROOT / "models" / "xnnpack"
TOK_DIR = REPO_ROOT / "models" / "tokenizers" / "smolvlm_500m_instruct"
MANIFEST_PATH = REPO_ROOT / "models" / "artifact_manifest.json"

# Fixed image shape for SigLIP encoder (from model config: image_size=512)
IMAGE_SIZE = 512
# Number of visual tokens after Idefics3Connector (1024 patches / scale_factor^2 = 1024/16 = 64)
N_VIS_TOKENS = 64
# Text hidden dim
HIDDEN_DIM = 960


# ── Wrapper modules ──────────────────────────────────────────────────────────

class FixedVisionEmbeddings(nn.Module):
    """
    Static-position-ID replacement for Idefics3VisionEmbeddings.

    The original implementation has a data-dependent loop:
        for batch_idx, p_attn_mask in enumerate(patch_attention_mask):
            nb_patches_h = p_attn_mask[:, 0].sum()   # data-dep tensor
            ...
            h_indices = torch.arange(nb_patches_h)    # symbolic shape!

    For a fixed 512×512 image (num_patches_per_side=32, all patches active),
    the fractional-coordinate + bucketize logic always yields position IDs
    equal to the standard row-major grid [0, 1, …, 1023].  We pre-register
    that buffer and skip the loop entirely.
    """
    def __init__(self, orig):
        super().__init__()
        self.patch_embedding = orig.patch_embedding
        self.position_embedding = orig.position_embedding
        n = orig.num_patches_per_side  # 32 for SmolVLM-500M
        self.register_buffer(
            "position_ids",
            torch.arange(n * n, dtype=torch.int64).unsqueeze(0),  # [1, 1024]
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        patch_embeds = self.patch_embedding(pixel_values)          # [B, D, 32, 32]
        embeddings = patch_embeds.flatten(2).transpose(1, 2)       # [B, 1024, D]
        ids = self.position_ids.expand(pixel_values.shape[0], -1)  # [B, 1024]
        return embeddings + self.position_embedding(ids)


class VisionEncoderModule(nn.Module):
    """
    Idefics3VisionTransformer + connector, export-safe for 512×512 images.

    Replaces Idefics3VisionEmbeddings (data-dependent arange loop) with
    FixedVisionEmbeddings (static position-ID buffer). Calls the encoder
    with attention_mask=None (full bidirectional attention — correct for
    full images with no padding patches).

    Attribute path in SmolVLM-500M-Instruct:
      model.model.vision_model  → Idefics3VisionTransformer
        .embeddings             → Idefics3VisionEmbeddings   (replaced)
        .encoder                → Idefics3Encoder
        .post_layernorm         → LayerNorm
      model.model.connector     → Idefics3Connector

    Input:  pixel_values  float32 [1, 3, 512, 512]
    Output: image_features float32 [1, 64, 960]
    """
    def __init__(self, base_model):
        super().__init__()
        vt = base_model.model.vision_model       # Idefics3VisionTransformer
        self.embeddings = FixedVisionEmbeddings(vt.embeddings)
        self.encoder = vt.encoder
        self.post_layernorm = vt.post_layernorm
        self.connector = base_model.model.connector

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        h = self.embeddings(pixel_values)                        # [1, 1024, 768]
        # Idefics3Encoder has no return_dict param — always returns BaseModelOutput
        h = self.encoder(inputs_embeds=h).last_hidden_state      # [1, 1024, 768]
        h = self.post_layernorm(h)
        return self.connector(h)                                 # [1, 64, 960]


class TokEmbeddingModule(nn.Module):
    """
    Token embedding table lookup.

    Input:  token_ids  int64 [1, seq_len]
    Output: embeds    float32 [1, seq_len, 960]
    """
    def __init__(self, base_model):
        super().__init__()
        self.embed = base_model.model.text_model.embed_tokens

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embed(token_ids)


class DecoderPrefillModule(nn.Module):
    """
    SmolLM2 transformer decoder — prefill / full-context mode.

    Takes pre-computed input embeddings (text + visual tokens merged by the
    caller) and returns logits over the vocabulary.

    Input:
      input_embeds  float32  [1, seq_len, 960]
      position_ids  int64    [1, seq_len]
    Output:
      logits        float32  [1, seq_len, 49280]

    No KV cache — caller re-runs the full sequence for each generated token.
    For latency-critical paths add a KV-cache decoder; for a demo this is
    simpler and sufficient.

    Builds an explicit causal mask internally. Passing attention_mask=None to
    LlamaDecoderLayer does NOT implicitly apply causal masking at this level
    (that construction normally happens in LlamaModel.forward, which this
    module bypasses) — verified empirically: without the mask, logits at an
    early position shifted when only a later/padded position's token changed
    (bidirectional leakage). Callers pad the real token sequence with trailing
    pad tokens up to seq_len; the causal mask ensures logits at the last real
    position are never contaminated by that padding.
    """
    def __init__(self, base_model):
        super().__init__()
        text = base_model.model.text_model
        self.layers = text.layers
        self.norm = text.norm
        self.lm_head = base_model.lm_head
        # In transformers 5.x RoPE is computed at model level, then passed
        # to each layer as (cos, sin) via position_embeddings kwarg.
        self.rotary_emb = text.rotary_emb

    def forward(
        self,
        input_embeds: torch.Tensor,   # [1, S, D]
        position_ids: torch.Tensor,   # [1, S]
    ) -> torch.Tensor:
        h = input_embeds
        seq_len = h.shape[1]
        causal_mask = torch.full((seq_len, seq_len), torch.finfo(h.dtype).min, dtype=h.dtype)
        causal_mask = torch.triu(causal_mask, diagonal=1).unsqueeze(0).unsqueeze(0)  # [1,1,S,S]
        # Pre-compute RoPE (cos, sin) for the full sequence once.
        position_embeddings = self.rotary_emb(h, position_ids)
        for layer in self.layers:
            # transformers 5.x: LlamaDecoderLayer returns torch.Tensor directly
            # (not a tuple) when output_attentions=False, use_cache=False.
            h = layer(
                hidden_states=h,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=None,
                output_attentions=False,
                use_cache=False,
                cache_position=None,
                position_embeddings=position_embeddings,
            )
        h = self.norm(h)
        return self.lm_head(h)


# ── Export helpers ───────────────────────────────────────────────────────────

def _export_to_pte(
    module: nn.Module,
    example_args: tuple,
    label: str = "",
) -> bytes:
    """
    torch.export (strict=True, fixed shapes) → to_edge_transform_and_lower
    (XNNPACK) → .pte bytes.

    We use strict=True to avoid symbolic-stride issues in ExecuTorch 1.3's
    spec_prop_pass.  All modules are exported with fixed shapes; callers must
    pad/slice inputs to match before invoking the runtime.
    """
    from torch.export import export
    from executorch.exir import to_edge_transform_and_lower, EdgeCompileConfig
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

    print(f"  [export]  tracing {label} (strict=True, fixed shapes) …")
    exported = export(module, example_args, strict=True)

    print(f"  [lower]   edge + XNNPACK partition for {label} …")
    edge = to_edge_transform_and_lower(
        exported,
        partitioner=[XnnpackPartitioner()],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )

    print(f"  [pte]     serialising {label} …")
    et_program = edge.to_executorch()
    return et_program.buffer


def _write_pte(buf: bytes, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buf)
    digest = hashlib.sha256(buf).hexdigest()
    mb = len(buf) / 1024 / 1024
    print(f"  wrote {path.name}  ({mb:.1f} MB)  sha256={digest[:16]}…")
    return digest


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--max-seq-len", type=int, default=128,
                   help="Fixed sequence length for decoder/embedding export (default 128)."
                        " Must be >= 64 visual tokens + max text prompt length."
                        " Caller pads inputs to exactly this length.")
    p.add_argument("--no-quantize", action="store_true",
                   help="Skip int8 weight quantization (export in float32)")
    p.add_argument("--cache-dir", default=None,
                   help="HuggingFace cache directory for model weights")
    return p.parse_args()


def load_model(cache_dir):
    from transformers import AutoModelForVision2Seq

    print(f"\nLoading {MODEL_ID} …")
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID,
        dtype=torch.float32,
        cache_dir=cache_dir,
        low_cpu_mem_usage=True,
        attn_implementation="eager",   # avoid flash-attn for clean export
    )
    model.eval()

    # Disable cache at config level so all `if use_cache:` branches
    # evaluate to False during tracing.
    model.config.use_cache = False
    model.config.text_config.use_cache = False
    if hasattr(model.model, "text_model"):
        model.model.text_model.config.use_cache = False

    return model


def _quantize_for_xnnpack(module: nn.Module, example_args: tuple, label: str = "") -> nn.Module:
    """
    Int8-quantize a module's nn.Linear layers via ExecuTorch's XNNPACKQuantizer + PT2E
    flow: per-channel int8 weights, dynamically-quantized int8 activations
    ("qd8-f32-qc8w" on XNNPACK). This is the flow XnnpackPartitioner actually lowers
    to efficient int8 kernels at inference time.

    torchao's `quantize_(model, Int8WeightOnlyConfig())` (the previous approach here)
    does NOT survive to_edge_transform_and_lower — its AffineQuantizedTensor subclass
    gets dequantized back to a plain float32 tensor by the subsequent `model.to(
    torch.float32)` call, so it was silently producing full fp32-sized .pte files
    despite logging "int8 quantization applied". Verified by comparing exported file
    size against the analytic fp32 parameter-count estimate — they matched.

    Dynamic (not static) activation quantization is used because it needs no
    calibration dataset — activation ranges are computed at runtime per forward call,
    which is safe for a single fixed-shape decoder/vision-encoder export like this.
    Only nn.Linear weights/activations are annotated; attention score math, RoPE, and
    norms stay float32.
    """
    from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
        XNNPACKQuantizer,
        get_symmetric_quantization_config,
    )
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    print(f"  [quantize] capturing {label} for int8 PT2E quantization …")
    captured = torch.export.export(module, example_args, strict=True).module()

    quantizer = XNNPACKQuantizer()
    quantizer.set_global(get_symmetric_quantization_config(is_per_channel=True, is_dynamic=True))
    prepared = prepare_pt2e(captured, quantizer)
    prepared(*example_args)  # single calibration pass; dynamic quant needs no real stats
    quantized = convert_pt2e(prepared)
    print(f"  [quantize] {label}: int8 dynamic-activation / per-channel weight quantization applied")
    return quantized


def export_vision_encoder(model, out_dir: Path, quantize: bool = True) -> str:
    print("\n=== Vision Encoder (SigLIP + Idefics3Connector) ===")
    mod = VisionEncoderModule(model).eval()
    dummy_pixels = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.float32)
    example_args = (dummy_pixels,)
    if quantize:
        mod = _quantize_for_xnnpack(mod, example_args, label="vision_encoder")
    buf = _export_to_pte(mod, example_args, label="vision_encoder")
    return _write_pte(buf, out_dir / "vision_encoder.pte")


def export_tok_embedding(model, out_dir: Path, max_seq_len: int) -> str:
    """
    Export with a FIXED seq_len = max_seq_len.
    Callers must pad token sequences to exactly max_seq_len with pad_id=0.

    Not quantized: this is a single nn.Embedding lookup (no nn.Linear), which
    XNNPACKQuantizer's default annotations don't cover. It stays float32
    (~189 MB for vocab 49280 x dim 960). A follow-up could hand-roll an int8
    embedding-table lookup (int8 rows + per-row scale) if this file's size
    becomes the binding constraint after the decoder/vision-encoder shrink.
    """
    print(f"\n=== Token Embedding (fixed seq_len={max_seq_len}) ===")
    mod = TokEmbeddingModule(model).eval()
    dummy_ids = torch.zeros(1, max_seq_len, dtype=torch.int64)
    buf = _export_to_pte(mod, (dummy_ids,), label="tok_embedding")
    return _write_pte(buf, out_dir / "tok_embedding.pte")


def export_decoder(model, out_dir: Path, max_seq_len: int, quantize: bool = True) -> str:
    """
    Export with a FIXED seq_len = max_seq_len.
    Callers must pad input_embeds to exactly max_seq_len and pass the
    position ids for the real (non-padded) positions in range [0, max_seq_len).
    The logit at the last real token position is used for greedy decoding.

    For autoregressive generation without KV cache the caller re-runs
    this module after appending each new token to the sequence (padding
    to max_seq_len each time).
    """
    print(f"\n=== Decoder — SmolLM2 prefill (fixed seq_len={max_seq_len}) ===")
    mod = DecoderPrefillModule(model).eval()
    dummy_embeds = torch.zeros(1, max_seq_len, HIDDEN_DIM, dtype=torch.float32)
    dummy_pos = torch.arange(max_seq_len, dtype=torch.int64).unsqueeze(0)
    example_args = (dummy_embeds, dummy_pos)
    if quantize:
        mod = _quantize_for_xnnpack(mod, example_args, label="smolvlm_decoder")
    buf = _export_to_pte(mod, example_args, label="smolvlm_decoder")
    return _write_pte(buf, out_dir / "smolvlm_decoder.pte")


def save_tokenizer(cache_dir, tok_dir: Path):
    from transformers import AutoProcessor
    print(f"\n=== Saving tokenizer to {tok_dir} ===")
    tok_dir.mkdir(parents=True, exist_ok=True)
    proc = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=cache_dir)
    proc.save_pretrained(str(tok_dir))
    print(f"  tokenizer saved ({len(list(tok_dir.iterdir()))} files)")


def update_manifest(sha_vision: str, sha_tok: str, sha_dec: str, out_dir: Path):
    """Add or update xnnpack artifact entries in artifact_manifest.json."""
    manifest = json.loads(MANIFEST_PATH.read_text())

    xnn_artifacts = [
        {
            "id": "smolvlm_vision_encoder_xnnpack",
            "task": "visual_question_answering",
            "runtime": "executorch_xnnpack",
            "path": str(out_dir.relative_to(REPO_ROOT) / "vision_encoder.pte"),
            "required": True,
            "sha256": sha_vision,
            "source": MODEL_ID,
            "notes": "SigLIP ViT + Idefics3Connector. XNNPACK (CPU) backend. "
                     "Input: [1,3,512,512] float32. Output: [1,64,960] float32.",
        },
        {
            "id": "smolvlm_tok_embedding_xnnpack",
            "task": "visual_question_answering",
            "runtime": "executorch_xnnpack",
            "path": str(out_dir.relative_to(REPO_ROOT) / "tok_embedding.pte"),
            "required": True,
            "sha256": sha_tok,
            "source": MODEL_ID,
            "notes": "Token embedding table. XNNPACK. "
                     "Input: [1, seq_len] int64. Output: [1, seq_len, 960] float32.",
        },
        {
            "id": "smolvlm_decoder_xnnpack",
            "task": "visual_question_answering",
            "runtime": "executorch_xnnpack",
            "path": str(out_dir.relative_to(REPO_ROOT) / "smolvlm_decoder.pte"),
            "required": True,
            "sha256": sha_dec,
            "source": MODEL_ID,
            "notes": "SmolLM2 decoder prefill, use_cache=False. XNNPACK. "
                     "Input: embeds [1,S,960] + pos_ids [1,S]. Output: logits [1,S,49280].",
        },
    ]

    # Remove stale xnnpack entries then append fresh ones
    existing = [a for a in manifest["artifacts"] if not a["id"].endswith("_xnnpack")]
    manifest["artifacts"] = existing + xnn_artifacts
    manifest["xnnpack_generated"] = True
    manifest["xnnpack_backend_note"] = (
        "XNNPACK artifacts run on any ARM CPU via ExecuTorch. "
        "For Hexagon NPU (SM8750) deploy use the QNN export path."
    )

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    print(f"\nManifest updated: {MANIFEST_PATH}")


def _skip_if_exists(path: Path, label: str):
    """Return cached sha256 if artifact already exists, else None."""
    if path.exists():
        print(f"  [skip] {label} already exists ({path.stat().st_size//1024//1024} MB)")
        return _sha256(path)
    return None


def main():
    args = parse_args()
    quantize = not args.no_quantize
    out_dir = OUT_DIR

    print(f"Target directory : {out_dir}")
    print(f"Max sequence len : {args.max_seq_len}")
    print(f"Quantize (int8)  : {quantize}")

    # ── Phase 1: vision encoder + token embedding ─────────────────────────
    # These need the full model (vision + text).
    sha_vision = _skip_if_exists(out_dir / "vision_encoder.pte", "vision_encoder")
    sha_tok    = _skip_if_exists(out_dir / "tok_embedding.pte",  "tok_embedding")

    if sha_vision is None or sha_tok is None:
        model = load_model(args.cache_dir)
        with torch.no_grad():
            if sha_vision is None:
                sha_vision = export_vision_encoder(model, out_dir, quantize=quantize)
            if sha_tok is None:
                sha_tok = export_tok_embedding(model, out_dir, args.max_seq_len)
        # Free vision model weights before decoder tracing to reclaim ~1.5 GB RAM.
        del model.model.vision_model
        del model.model.connector
        import gc; gc.collect()
        print("\n  [mem] freed vision model and connector")
    else:
        model = load_model(args.cache_dir)
        del model.model.vision_model
        del model.model.connector
        import gc; gc.collect()

    # ── Phase 2: decoder ──────────────────────────────────────────────────
    sha_dec = _skip_if_exists(out_dir / "smolvlm_decoder.pte", "smolvlm_decoder")
    if sha_dec is None:
        with torch.no_grad():
            sha_dec = export_decoder(model, out_dir, args.max_seq_len, quantize=quantize)

    del model
    import gc; gc.collect()

    save_tokenizer(args.cache_dir, TOK_DIR)
    update_manifest(sha_vision, sha_tok, sha_dec, out_dir)

    print("\n✓ Export complete.")
    print(f"  {out_dir}/vision_encoder.pte")
    print(f"  {out_dir}/tok_embedding.pte")
    print(f"  {out_dir}/smolvlm_decoder.pte")
    print(f"  {TOK_DIR}/")
    print("\nTo verify artifacts load correctly:")
    print("  python scripts/verify_pte.py")
    print("\nTo target Hexagon NPU instead, run:")
    print("  QNN_SDK_ROOT=/path/to/qairt DEVICE_SERIAL=<adb> bash scripts/setup_executorch.sh")


if __name__ == "__main__":
    main()
