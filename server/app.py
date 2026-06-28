import json
import os
import queue
import tempfile
import threading
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from server.memory import (
    init_db, list_memories, delete_memory, delete_all_memories,
    retrieve_memories, retrieve_memories_by_image,
    load_settings, save_settings,
)
from server.pipeline import (
    transcribe_audio, ask_vlm_stream, scene_describe, speak, save_pipeline,
)

app = Flask(__name__)
CORS(app)

init_db()

# Keywords that route a voice query to save instead of generate
_SAVE_VERBS    = frozenset({'save', 'remember', 'store', 'memorize', 'note', 'keep'})
_SAVE_PREFIXES = ('save ', 'remember ', 'store ', 'memorize ', 'note that ', 'keep this ')
_SAVE_PHRASES  = ('add to memory', "don't forget", 'take note')


def _is_save_intent(text):
    lower = text.lower().strip()
    return (
        any(lower.startswith(p) for p in _SAVE_PREFIXES) or
        any(p in lower for p in _SAVE_PHRASES)
    )


def _extract_save_content(text):
    """Strip ONLY the trigger verb, keeping everything else verbatim.

    e.g. 'remember us, this is Alekya and Sarah'
         -> 'us, this is Alekya and Sarah'
         'note that Alekya works here'
         -> 'that Alekya works here'
    """
    stripped = text.strip()
    # Fast path: check if first word is a known save verb
    first_space = stripped.find(' ')
    if first_space != -1:
        verb = stripped[:first_space].lower().rstrip(',.!')
        if verb in _SAVE_VERBS:
            return stripped[first_space + 1:].strip()
    # Phrase-in-middle fallback ("don't forget", "add to memory", etc.)
    lower = stripped.lower()
    for phrase in _SAVE_PHRASES:
        if phrase in lower:
            idx = lower.index(phrase) + len(phrase)
            return stripped[idx:].strip().lstrip(':').strip() or stripped
    return stripped


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/ready")
def ready():
    from server.pipeline import smolvlm_model, embed_model, clip_model
    from server.runtime_config import runtime_readiness
    vlm_ready  = smolvlm_model is not None
    embed_ready = embed_model is not None
    clip_ready  = clip_model is not None
    on_device = runtime_readiness()
    return jsonify({
        "vlm_ready":   vlm_ready,
        "embed_ready": embed_ready,
        "clip_ready":  clip_ready,
        "ready":       vlm_ready and clip_ready,
        "on_device_runtime": on_device,
    })


@app.route("/query", methods=["POST"])
def query():
    """Streaming SSE endpoint — NO auto-save. Only answers queries."""
    # Read all request data upfront before streaming begins
    question_text = (request.form.get("question") or "").strip()

    audio_data = audio_ext = None
    audio_file = request.files.get("audio")
    if audio_file:
        audio_data = audio_file.read()
        ct = audio_file.content_type or ""
        audio_ext = ".mp4" if "mp4" in ct or "m4a" in ct else ".webm"

    image_data = image_ext = None
    image_file = request.files.get("image")
    if image_file:
        image_data = image_file.read()
        image_ext = os.path.splitext(image_file.filename or ".jpg")[1] or ".jpg"

    ev = queue.Queue()

    def worker():
        tmp_audio = tmp_image = None
        try:
            question = question_text

            if audio_data:
                with tempfile.NamedTemporaryFile(suffix=audio_ext, delete=False) as f:
                    f.write(audio_data)
                    tmp_audio = f.name
                ev.put(("status", "Transcribing audio..."))
                question = transcribe_audio(tmp_audio)

            if not question:
                ev.put(("error", "No question provided (add audio or question field)"))
                return

            # Write image to a temp file BEFORE the save-routing check so the
            # image is available whether we're saving or querying.
            if image_data:
                with tempfile.NamedTemporaryFile(suffix=image_ext, delete=False) as f:
                    f.write(image_data)
                    tmp_image = f.name

            # Voice save routing — detect save intent and short-circuit to save_pipeline
            if _is_save_intent(question):
                content = _extract_save_content(question)
                ev.put(("status", "Saving to memory..."))
                save_pipeline(content, tag="voice", image_path=tmp_image, compress=False)
                ev.put(("saved", content))
                return

            ev.put(("status", "Searching memory..."))

            # Text-based retrieval
            text_memories = retrieve_memories(question)

            # Visual retrieval — CLIP (30ms) identifies who/what is in frame.
            # This is what makes "who is this?" work even with no keyword overlap.
            visual_memories = []
            if tmp_image:
                try:
                    from server.pipeline import get_clip_embedding, clip_model
                    from server.memory import retrieve_memories_by_image as _rmi
                    if clip_model is not None:
                        vis_emb    = get_clip_embedding(tmp_image)
                        vis_result = _rmi(vis_emb)
                        if vis_result:
                            vis_dict, _ = vis_result
                            vis_text = vis_dict.get("text", "").split("\n[Visual:")[0].strip()
                            if vis_text:
                                visual_memories = [vis_text]
                except Exception:
                    pass  # non-fatal

            # Visual context goes first — it tells the VLM WHO/WHAT it's looking at.
            seen = set(visual_memories)
            memories = visual_memories + [m for m in text_memories if m not in seen]
            memories = memories[:3]

            ev.put(("status", "Generating answer..."))
            full_answer = ""
            for token in ask_vlm_stream(question, image_path=tmp_image, context=memories):
                full_answer += token
                ev.put(("token", token))

            ev.put(("done", (full_answer, len(memories))))

            if full_answer:
                threading.Thread(target=speak, args=(full_answer,), daemon=True).start()

        except Exception as e:
            ev.put(("error", str(e)))
        finally:
            if tmp_audio and os.path.exists(tmp_audio):
                os.unlink(tmp_audio)
            if tmp_image and os.path.exists(tmp_image):
                os.unlink(tmp_image)
            ev.put(None)

    threading.Thread(target=worker, daemon=True).start()

    def generate():
        while True:
            item = ev.get()
            if item is None:
                break
            event_type, payload = item
            if event_type == "status":
                yield f"data: {json.dumps({'type': 'status', 'message': payload})}\n\n"
            elif event_type == "token":
                yield f"data: {json.dumps({'type': 'token', 'text': payload})}\n\n"
            elif event_type == "done":
                answer, mem_count = payload
                yield f"data: {json.dumps({'type': 'done', 'answer': answer, 'memory_count': mem_count})}\n\n"
            elif event_type == "saved":
                yield f"data: {json.dumps({'type': 'saved', 'text': payload})}\n\n"
            elif event_type == "error":
                yield f"data: {json.dumps({'type': 'error', 'message': payload})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.route("/query_simple", methods=["POST"])
def query_simple():
    """Simple non-streaming query. Calls retrieve_memories + ask_vlm_stream. NEVER saves."""
    try:
        question     = (request.form.get("question") or "").strip()
        yolo_context = (request.form.get("yolo_context") or "").strip()
        image        = request.files.get("image")
        audio        = request.files.get("audio")

        if audio:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                audio.save(f.name)
                tmp_audio = f.name
            question = transcribe_audio(tmp_audio)
            os.unlink(tmp_audio)

        # Prepend YOLO detection context so the VLM knows what was on-screen
        if yolo_context:
            enhanced_question = f"The camera detects: {yolo_context}. Question: {question}"
        else:
            enhanced_question = question

        image_path = None
        if image:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                image.save(f.name)
                image_path = f.name

        memories = retrieve_memories(enhanced_question) if enhanced_question else []
        answer = "".join(ask_vlm_stream(enhanced_question, image_path=image_path, context=memories))

        if image_path and os.path.exists(image_path):
            os.unlink(image_path)

        return jsonify({
            "question":     question,
            "answer":       answer,
            "yolo_context": yolo_context,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/save", methods=["POST"])
def save():
    """The ONLY way to save a memory. Accepts JSON body OR multipart form with optional image."""
    try:
        if request.content_type and "application/json" in request.content_type:
            data = request.get_json(force=True)
            text = (data.get("text") or "").strip()
            tag  = (data.get("tag") or "").strip()
            image_path = None
        else:
            text  = (request.form.get("text") or "").strip()
            tag   = (request.form.get("tag") or "").strip()
            image = request.files.get("image")
            image_path = None
            if image:
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                    image.save(f.name)
                    image_path = f.name

        if not text:
            return jsonify({"error": "text required"}), 400

        result = save_pipeline(text, tag=tag, image_path=image_path, compress=False)

        if image_path and os.path.exists(image_path):
            os.unlink(image_path)

        return jsonify({"result": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/memories", methods=["GET"])
def memories():
    """Return memories with total, limit, usage_percent, and settings."""
    try:
        mems     = list_memories()
        settings = load_settings()
        max_mem  = settings.get("max_memories", 500)
        total    = len(mems)
        usage_pct = int(round(total / max_mem * 100)) if max_mem > 0 else 0
        return jsonify({
            "memories":      mems,
            "total":         total,
            "limit":         max_mem,
            "usage_percent": usage_pct,
            "settings": {
                "max_memories":         settings.get("max_memories", 500),
                "similarity_threshold": settings.get("similarity_threshold", 0.3),
                "auto_cleanup":         settings.get("auto_cleanup", True),
            },
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/memories/<int:memory_id>", methods=["DELETE"])
def delete(memory_id):
    try:
        delete_memory(memory_id)
        return jsonify({"result": "deleted"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/memories/all", methods=["DELETE"])
def delete_all():
    try:
        delete_all_memories()
        return jsonify({"result": "all memories deleted"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/memories/image/<int:memory_id>", methods=["GET"])
def memory_image(memory_id):
    """Return the stored JPEG for a memory, or 404 if none."""
    from flask import send_file, abort
    from server.memory import DB_PATH
    import sqlite3 as _sqlite3
    try:
        conn = _sqlite3.connect(DB_PATH)
        row  = conn.execute(
            "SELECT image_path FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()
        conn.close()
        if not row or not row[0] or not os.path.exists(row[0]):
            abort(404)
        return send_file(row[0], mimetype="image/jpeg")
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/settings", methods=["GET", "POST"])
def settings():
    """Get or update settings (max_memories, similarity_threshold, auto_cleanup)."""
    try:
        if request.method == "GET":
            return jsonify(load_settings())
        else:
            data = request.get_json(force=True)
            current = load_settings()
            # Merge in only recognised keys
            for key in ("max_memories", "auto_cleanup", "similarity_threshold", "visual_threshold"):
                if key in data:
                    current[key] = data[key]
            save_settings(current)
            return jsonify({"result": "settings updated", "settings": current})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/scene", methods=["POST"])
def scene():
    """Auto-describe what the camera sees + surface any matching memories.

    Form fields:
      image  — JPEG frame (optional but recommended)
      labels — comma-separated YOLO labels, e.g. "person,laptop,chair"
    Response:
      { description, matches, save_suggested, labels }
    """
    try:
        yolo_labels = (request.form.get("labels") or "").strip()
        image       = request.files.get("image")

        image_path = None
        if image:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                image.save(f.name)
                image_path = f.name

        # Build search query from labels + "person" flag
        label_list = [l.strip() for l in yolo_labels.split(",") if l.strip()]
        has_person = any(l in ("person", "face") for l in label_list)
        search_text = " ".join(label_list) if label_list else ""

        # Retrieve memories relevant to what's in frame
        matches = retrieve_memories(search_text, top_k=3) if search_text else []

        # Describe the scene weaving in known context
        description = scene_describe(image_path, yolo_labels, matches)

        if image_path and os.path.exists(image_path):
            os.unlink(image_path)

        # Suggest saving when a person is in frame but none of the matches mention a name
        person_known = any(
            any(word in m.lower() for word in ["name", "is", "called", "this is"])
            for m in matches
        )
        save_suggested = has_person and not person_known

        return jsonify({
            "description":    description,
            "matches":        matches,
            "save_suggested": save_suggested,
            "labels":         label_list,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/save_scene", methods=["POST"])
def save_scene():
    """Save the current scene with an image for face/object memory.

    Form fields:
      text  — label/note for this scene (e.g. "this is Sarah, my friend")
      image — JPEG frame to store alongside the text
      tag   — optional tag (default "scene")
    """
    try:
        text  = (request.form.get("text") or "").strip()
        tag   = (request.form.get("tag") or "scene").strip()
        image = request.files.get("image")

        if not text:
            return jsonify({"error": "text is required"}), 400

        image_path = None
        if image:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                image.save(f.name)
                image_path = f.name

        result = save_pipeline(text, tag=tag, image_path=image_path, compress=False)

        if image_path and os.path.exists(image_path):
            os.unlink(image_path)

        return jsonify({"result": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/recognize", methods=["POST"])
def recognize():
    """Passive recognition via CLIP — no VLM involved, fast (~30ms embedding)."""
    try:
        from server.pipeline import get_clip_embedding_from_bytes, clip_model
        image = request.files.get("image")
        if not image:
            return jsonify({"match": False})
        if clip_model is None:
            return jsonify({"match": False, "reason": "CLIP not loaded yet"})

        image_bytes = image.read()
        query_emb   = get_clip_embedding_from_bytes(image_bytes)
        result      = retrieve_memories_by_image(query_emb)

        if result:
            memory, confidence = result
            # Return only what the user saved — strip the AI image description.
            raw_text     = memory.get("text", "")
            display_text = raw_text.split("\n[Visual:")[0].strip()
            return jsonify({
                "match":        True,
                "memory":       memory,
                "display_text": display_text,
                "confidence":   float(confidence),
            })
        return jsonify({"match": False})
    except Exception as e:
        return jsonify({"match": False, "error": str(e)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
