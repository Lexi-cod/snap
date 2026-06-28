import os
import json
import tempfile
import threading
import wave
import torch
from PIL import Image
from faster_whisper import WhisperModel
from server.memory import init_db, retrieve_memories, save_memory, update_memory_image

INTERNVL_REPO = "OpenGVLab/InternVL3-1B"
PIPER_MODEL   = os.path.expanduser("~/snapon/data/piper/en_US-lessac-medium.onnx")

SYSTEM_PROMPT = """You are a personal visual memory assistant named SnapOn.
- When saving: confirm warmly in one sentence what you saved
- When answering: be direct, personal, 1-2 sentences max
- Use saved memory context naturally
- Never say 'As an AI'
- Answer from general knowledge if no memory matches
- Only say 'I don't have that saved' for personal facts"""

# Public globals — populated by _preload_all() in background thread.
smolvlm_model     = None
smolvlm_processor = None
embed_model       = None

# CLIP globals — used exclusively for visual similarity (recognition).
# InternVL3 is NOT used for recognition; CLIP is faster and purpose-built.
clip_model      = None
clip_preprocess = None

_whisper  = WhisperModel("base.en", device="cpu", compute_type="int8")
_vlm_lock = threading.Lock()
_piper_voice = None

_INTERN_MEAN = (0.485, 0.456, 0.406)
_INTERN_STD  = (0.229, 0.224, 0.225)
_INTERN_SIZE = 448


def load_clip():
    """Load CLIP ViT-B-32 (openai weights) for visual similarity search."""
    global clip_model, clip_preprocess
    import open_clip
    import numpy as np
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    clip_model.eval()
    print("[CLIP] ViT-B-32 ready — 512-dim visual embeddings")


def get_clip_embedding(image_path: str) -> "np.ndarray":
    """CLIP visual embedding for a saved image file — 512-dim, L2-normalised."""
    import numpy as np
    if clip_model is None or clip_preprocess is None:
        raise RuntimeError("CLIP not loaded")
    img = Image.open(image_path).convert("RGB")
    tensor = clip_preprocess(img).unsqueeze(0)
    with torch.no_grad():
        feat = clip_model.encode_image(tensor)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.squeeze().cpu().numpy().astype(np.float32)


def get_clip_embedding_from_bytes(image_bytes: bytes) -> "np.ndarray":
    """CLIP visual embedding from raw JPEG/PNG bytes — used by /recognize."""
    import io
    import numpy as np
    if clip_model is None or clip_preprocess is None:
        raise RuntimeError("CLIP not loaded")
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = clip_preprocess(img).unsqueeze(0)
    with torch.no_grad():
        feat = clip_model.encode_image(tensor)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.squeeze().cpu().numpy().astype(np.float32)


def _intern_preprocess(image_path: str) -> torch.Tensor:
    """Single-tile 448x448 preprocessing for InternVL3 (fast enough for CPU)."""
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode
    transform = T.Compose([
        T.Resize((_INTERN_SIZE, _INTERN_SIZE), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=_INTERN_MEAN, std=_INTERN_STD),
    ])
    img = Image.open(image_path).convert("RGB")
    return transform(img).unsqueeze(0).to(torch.bfloat16)  # [1, 3, 448, 448]


def _get_vlm():
    global smolvlm_processor, smolvlm_model
    with _vlm_lock:
        if smolvlm_model is None:
            import time
            from transformers import AutoModel, AutoTokenizer
            import transformers.modeling_utils as _mu

            # transformers 5.x calls self.all_tied_weights_keys in
            # _move_missing_keys_from_meta_to_device, but InternVLChatModel
            # (trust_remote_code) never sets it. Patch the method to add a
            # default before it's accessed — only when the attribute is absent.
            _orig_move = _mu.PreTrainedModel._move_missing_keys_from_meta_to_device
            def _patched_move(self, *args, **kwargs):
                if not hasattr(self, "all_tied_weights_keys"):
                    self.all_tied_weights_keys = {}
                return _orig_move(self, *args, **kwargs)
            _mu.PreTrainedModel._move_missing_keys_from_meta_to_device = _patched_move

            print("[internvl] loading InternVL3-1B...")
            t0 = time.time()
            smolvlm_processor = AutoTokenizer.from_pretrained(
                INTERNVL_REPO, trust_remote_code=True, use_fast=False
            )

            try:
                from transformers import BitsAndBytesConfig
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
                smolvlm_model = AutoModel.from_pretrained(
                    INTERNVL_REPO,
                    quantization_config=quant_config,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                ).eval()
                print("[internvl] loaded with INT4")
            except Exception as e:
                print(f"[internvl] INT4 failed ({e}), using bfloat16")
                smolvlm_model = AutoModel.from_pretrained(
                    INTERNVL_REPO,
                    dtype=torch.bfloat16,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                ).eval()

            print(f"[internvl] ready — loaded in {time.time() - t0:.1f}s")
    return smolvlm_processor, smolvlm_model


def _get_piper_voice():
    global _piper_voice
    if _piper_voice is not None:
        return _piper_voice
    if not os.path.exists(PIPER_MODEL):
        return None
    try:
        from piper.voice import PiperVoice
        _piper_voice = PiperVoice.load(PIPER_MODEL)
        return _piper_voice
    except Exception as e:
        print(f"[speak] piper load failed: {e}")
        return None


def post_process(text: str) -> str:
    """Strip AI disclaimers and markdown; ensure clean capitalization and punctuation."""
    for prefix in [
        "As an AI", "I'm an AI", "As a language model",
        "I cannot", "I don't have access", "Unfortunately",
        "I apologize", "I am an AI",
    ]:
        if text.strip().startswith(prefix):
            parts = text.split('. ', 1)
            text = parts[1] if len(parts) > 1 else text

    text = text.replace('**', '').replace('*', '').replace('_', '')
    text = text.replace('```', '').replace('`', '')
    text = text.replace('#', '').replace('>', '')

    text = ' '.join(text.split()).strip()

    if text:
        text = text[0].upper() + text[1:]
    if text and text[-1] not in '.!?':
        text += '.'

    return text


def compress_text(text: str) -> str:
    """Summarize long text to one sentence via InternVL3."""
    if len(text.split()) < 20:
        return text
    try:
        tokenizer, model = _get_vlm()
        prompt = f"Summarize in one concise sentence preserving all key facts:\n{text}\nSummary:"
        gen_cfg = dict(max_new_tokens=50, do_sample=True, temperature=0.3)
        with torch.no_grad():
            result = model.chat(tokenizer, None, prompt, gen_cfg)
        if isinstance(result, tuple):
            result = result[0]
        return post_process(result.strip())
    except Exception:
        return text


def transcribe_audio(audio_path):
    segments, _ = _whisper.transcribe(audio_path)
    return " ".join(seg.text.strip() for seg in segments).strip()


def speak(text):
    """Synthesize and play text via piper-tts. Silent-fails if unavailable."""
    voice = _get_piper_voice()
    if voice is None:
        print("[speak] piper model not ready — TTS skipped")
        return
    try:
        import sounddevice as sd
        import scipy.io.wavfile as wavfile

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            with wave.open(tmp_path, "wb") as wf:
                voice.synthesize(text, wf)
            rate, data = wavfile.read(tmp_path)
            sd.play(data, rate)
            sd.wait()
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    except Exception as e:
        print(f"[speak] playback error: {e}")


def embed_text(text: str):
    """Embed text using InternVL3's LLM backbone (Qwen2) — 896-dim output."""
    import numpy as np
    import torch.nn.functional as F

    tokenizer, vlm = _get_vlm()
    enc = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=256, padding=True
    )
    with torch.no_grad():
        # InternVLChatModel.language_model = Qwen2ForCausalLM
        # .model strips the LM head -> Qwen2Model -> outputs last_hidden_state
        lm   = getattr(vlm, "language_model", vlm)
        base = getattr(lm, "model", lm)
        out  = base(**enc)
        last = out.last_hidden_state.float()          # [1, seq, hidden_dim]
        mask = enc["attention_mask"].unsqueeze(-1).float()
        emb  = (last * mask).sum(1) / mask.sum(1).clamp(min=1)
    emb = F.normalize(emb, p=2, dim=-1)
    return emb.squeeze(0).cpu().numpy().astype("float32")


def get_image_embedding(image_path: str):
    """
    Get a 896-dim embedding for an image using InternVL3.

    Steps:
      1. Preprocess image to 448x448 tensor
      2. Run through model.vision_model to get ViT features [1, N, 1024]
      3. Mean pool -> [1, 1024]
      4. Project via model.mlp1 -> [1, 896]  (ViT -> language space)
      5. L2 normalize, return 896-dim np.float32

    Falls back to running through language model if mlp1 doesn't exist or
    dimension doesn't match.
    """
    import numpy as np
    import torch.nn.functional as F

    tokenizer, vlm = _get_vlm()

    try:
        pv = _intern_preprocess(image_path)  # [1, 3, 448, 448]

        with torch.no_grad():
            vision_model = getattr(vlm, "vision_model", None)
            mlp1 = getattr(vlm, "mlp1", None)

            if vision_model is not None:
                # Run through ViT
                vision_out = vision_model(pv)
                # Extract features: handle both BaseModelOutput and tuple
                if hasattr(vision_out, "last_hidden_state"):
                    vit_features = vision_out.last_hidden_state  # [1, N, 1024]
                elif isinstance(vision_out, tuple):
                    vit_features = vision_out[0]
                else:
                    vit_features = vision_out

                # Mean pool over sequence dimension
                img_emb = vit_features.mean(dim=1)  # [1, 1024]

                if mlp1 is not None:
                    try:
                        # Project from ViT space (1024) to language space (896)
                        projected = mlp1(img_emb.float())  # [1, 896]
                        if projected.shape[-1] == 896:
                            emb = F.normalize(projected, p=2, dim=-1)
                            return emb.squeeze(0).cpu().numpy().astype("float32")
                        else:
                            print(f"[get_image_embedding] mlp1 output dim {projected.shape[-1]} != 896, falling back")
                    except Exception as e:
                        print(f"[get_image_embedding] mlp1 projection failed: {e}, falling back")

                # If mlp1 failed or dim mismatch: project via a simple linear
                # operation or fall through to language model fallback
                # First try: if ViT features are already 896 (unlikely but handle)
                if img_emb.shape[-1] == 896:
                    emb = F.normalize(img_emb.float(), p=2, dim=-1)
                    return emb.squeeze(0).cpu().numpy().astype("float32")

        # Fallback: describe the image as text, then embed the description
        print("[get_image_embedding] falling back to text embedding of image description")
        desc = describe_image(image_path)
        return embed_text(desc)

    except Exception as e:
        print(f"[get_image_embedding] failed ({e}), using zero vector")
        import numpy as np
        return np.zeros(896, dtype="float32")


def ask_vlm_stream(question, image_path=None, context=None):
    """Generator yielding response tokens via InternVL3 for all queries (vision + text)."""
    tokenizer, model = _get_vlm()

    memory_context = "\n".join(f"- {m}" for m in context) if context else ""

    if memory_context:
        prompt = f"{SYSTEM_PROMPT}\n\nPersonal context:\n{memory_context}\n\nQuestion: {question}\nAnswer:"
    else:
        prompt = f"{SYSTEM_PROMPT}\n\nQuestion: {question}\nAnswer:"

    gen_cfg = dict(
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.3,
    )

    with torch.no_grad():
        if image_path:
            pv = _intern_preprocess(image_path)
            response = model.chat(tokenizer, pv, f"<image>\n{prompt}", gen_cfg)
        else:
            response = model.chat(tokenizer, None, prompt, gen_cfg)

    if isinstance(response, tuple):
        response = response[0]

    full = post_process(response.strip())

    # Safety-alignment workaround: if VLM ignored provided context, surface top memory directly
    if context and "don't have that" in full.lower():
        raw = context[0].strip()
        if not raw.endswith((".", "!", "?")):
            raw += "."
        full = raw[0].upper() + raw[1:]

    words = full.split(" ")
    for i, word in enumerate(words):
        yield word if i == len(words) - 1 else word + " "


def describe_image(image_path):
    """Run InternVL3 on an image and return a brief scene/people description."""
    tokenizer, model = _get_vlm()
    pv = _intern_preprocess(image_path)
    gen_cfg = dict(max_new_tokens=80, do_sample=False)
    with torch.no_grad():
        response = model.chat(
            tokenizer, pv,
            "<image>\nBriefly describe the people and scene visible in this image.",
            gen_cfg,
        )
    if isinstance(response, tuple):
        response = response[0]
    return response.strip()


def scene_describe(image_path=None, detected_labels="", matched_memories=None):
    """Describe the current camera view naturally, weaving in known context."""
    tokenizer, model = _get_vlm()

    parts = [SYSTEM_PROMPT]
    if matched_memories:
        known = "; ".join(matched_memories[:2])
        parts.append(f"\nPeople/things I know about: {known}")
    if detected_labels:
        parts.append(f"\nObjects visible (YOLO): {detected_labels}")
    parts.append("\n\nDescribe what you see:")
    prompt = "".join(parts)

    gen_cfg = dict(max_new_tokens=80, do_sample=True, temperature=0.6, top_p=0.9, repetition_penalty=1.2)

    with torch.no_grad():
        if image_path:
            pv = _intern_preprocess(image_path)
            response = model.chat(tokenizer, pv, f"<image>\n{prompt}", gen_cfg)
        else:
            response = model.chat(tokenizer, None, prompt, gen_cfg)

    if isinstance(response, tuple):
        response = response[0]
    return post_process(response.strip())


def save_pipeline(text, tag="", image_path=None, compress=False):
    """
    Save a memory and return a confirmation string.
    This is the ONLY place saves happen — never called from query functions.
    """
    import shutil
    import uuid

    stored_image_path = None

    if image_path and os.path.exists(image_path):
        images_dir = os.path.expanduser("~/snapon/data/images")
        os.makedirs(images_dir, exist_ok=True)
        ext = os.path.splitext(image_path)[1] or ".jpg"
        stored_image_path = os.path.join(images_dir, f"{uuid.uuid4().hex}{ext}")
        shutil.copy2(image_path, stored_image_path)

    # Save text immediately — user sees confirmation in <1s.
    # compress=False for voice saves (verbatim); compress=True if explicitly requested.
    stored_text, memory_id = save_memory(
        text, tag,
        image_path=stored_image_path,
        image_description=None,
        compress=compress,
    )

    # InternVL3 image description runs in background using the permanent copy.
    if stored_image_path and memory_id:
        def _bg_describe(mid=memory_id, img=stored_image_path):
            try:
                desc = describe_image(img)
                print(f"[save] image described (bg, id={mid}): {desc}")
                update_memory_image(mid, desc)
            except Exception as e:
                print(f"[save] bg image description failed (id={mid}): {e}")
        threading.Thread(target=_bg_describe, daemon=True).start()

    return f"Saved: {stored_text}"


def _preload_all():
    global embed_model
    # ── InternVL3-1B (text embedding + VLM answering) ──────────────────────
    try:
        _get_vlm()
        embed_model = embed_text
        print("[pipeline] InternVL3 ready — text embeddings via Qwen2 backbone")
        from server.memory import TEXT_INDEX_PATH, _rebuild_text_index
        if not os.path.exists(TEXT_INDEX_PATH):
            print("[pipeline] Text FAISS index missing — rebuilding...")
            _rebuild_text_index()
    except Exception as e:
        print(f"[internvl] background pre-load failed: {e}")

    # ── CLIP ViT-B-32 (visual similarity for /recognize) ───────────────────
    try:
        load_clip()
        from server.memory import VISUAL_INDEX_PATH, _rebuild_visual_index
        if not os.path.exists(VISUAL_INDEX_PATH):
            print("[pipeline] Visual FAISS index missing — rebuilding with CLIP...")
            _rebuild_visual_index()
    except Exception as e:
        print(f"[CLIP] background load failed: {e}")


threading.Thread(target=_preload_all, daemon=True).start()
