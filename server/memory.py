import json
import os
import re
import sqlite3
import threading
import numpy as np
import faiss
import torch
from datetime import datetime

DB_PATH   = os.path.expanduser("~/snapon/data/memory.db")

# Text FAISS index — InternVL3/Qwen2 embeddings, 896-dim, all memories.
TEXT_INDEX_PATH  = os.path.expanduser("~/snapon/data/memory_text.index")
EMBEDDING_DIM    = 896

# Visual FAISS index — CLIP ViT-B-32, 512-dim, image memories only.
# Uses IndexIDMap so search returns actual SQLite memory IDs, not positional offsets.
VISUAL_INDEX_PATH = os.path.expanduser("~/snapon/data/memory_visual.index")
VISUAL_DIM        = 512

# Backward-compat alias used in older pipeline.py imports (updated in _preload_all).
INDEX_PATH = TEXT_INDEX_PATH

SIMILARITY_THRESHOLD  = 0.3
PERSON_THRESHOLD      = 0.2
RERANK_SKIP_THRESHOLD = 0.50
MAX_MEMORIES          = 500
AUTO_CLEANUP          = True
SETTINGS_FILE         = os.path.expanduser("~/snapon/data/settings.json")

_STOPWORDS = frozenset({
    'a','an','the','is','are','was','were','be','been','being',
    'have','has','had','do','does','did','will','would','could','should',
    'i','my','your','their','our','its','this','that','these','those',
    'what','where','when','who','how','why','which',
    'in','on','at','to','of','for','with','by','from','up','about','into',
    'and','or','but','if','as','so','yet',
})

_PERSON_KEYWORDS = frozenset({
    'who', 'person', 'people', 'face', 'faces', 'man', 'woman', 'girl', 'boy',
    'name', 'identify', 'recognize', 'know', 'these', 'those', 'they',
})

_save_counter = 0


# ── Settings ──────────────────────────────────────────────────────────────────

def load_settings() -> dict:
    defaults = {
        "max_memories": 500,
        "auto_cleanup": True,
        "similarity_threshold": 0.3,
        "visual_threshold": 0.72,
    }
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE) as f:
                defaults.update(json.load(f))
        except Exception:
            pass
    return defaults


def save_settings(settings: dict):
    os.makedirs(os.path.dirname(SETTINGS_FILE), exist_ok=True)
    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings, f, indent=2)


# ── Keyword scoring ───────────────────────────────────────────────────────────

def _keyword_score(query: str, memory: str) -> float:
    """Fraction of non-trivial query terms found in memory (prefix stemming)."""
    def kw(text):
        return [w for w in re.findall(r'\b\w+\b', text.lower())
                if w not in _STOPWORDS and len(w) > 2]
    qwords = kw(query)
    if not qwords:
        return 0.0
    mset = set(kw(memory))
    matched = 0.0
    for qw in qwords:
        if qw in mset:
            matched += 1.0
        elif len(qw) >= 4 and any(mw.startswith(qw[:4]) for mw in mset):
            matched += 0.8
    return matched / len(qwords)


def _is_person_query(query):
    words = set(query.lower().split())
    return bool(words & _PERSON_KEYWORDS)


# ── Embedding helpers (InternVL3 text / old image — text index only) ──────────

def get_embedding(text: str) -> np.ndarray:
    from server.pipeline import embed_text
    return embed_text(text).astype(np.float32)


def get_text_embedding(text: str) -> np.ndarray:
    return get_embedding(text)


# ── Text FAISS index (896-dim InternVL3, all memories) ────────────────────────

def _load_text_index():
    if os.path.exists(TEXT_INDEX_PATH):
        idx = faiss.read_index(TEXT_INDEX_PATH)
        if idx.d != EMBEDDING_DIM:
            print(f"[memory] Text FAISS dim mismatch ({idx.d} vs {EMBEDDING_DIM}), rebuilding...")
            os.remove(TEXT_INDEX_PATH)
            return _rebuild_text_index()
        return idx
    return faiss.IndexFlatIP(EMBEDDING_DIM)


def _save_text_index(index):
    os.makedirs(os.path.dirname(TEXT_INDEX_PATH), exist_ok=True)
    faiss.write_index(index, TEXT_INDEX_PATH)


def _rebuild_text_index():
    """Rebuild 896-dim text FAISS index using InternVL3 text embeddings."""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT text FROM memories ORDER BY id").fetchall()
    conn.close()
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    for i, (text,) in enumerate(rows):
        vec = get_embedding(text)
        index.add(vec.reshape(1, -1))
        if (i + 1) % 10 == 0:
            print(f"[memory] text index rebuild: {i + 1}/{len(rows)}")
    _save_text_index(index)
    print(f"[memory] text index rebuilt: {len(rows)} memories at {EMBEDDING_DIM}-dim")
    return index


# Backward-compat alias used by older code paths.
def _rebuild_index_from_db():
    return _rebuild_text_index()


# ── Visual FAISS index (512-dim CLIP, image memories only, IndexIDMap) ────────

def _load_visual_index():
    """Load or create the CLIP visual index. Returns an IndexIDMap(IndexFlatIP(512))."""
    if os.path.exists(VISUAL_INDEX_PATH):
        try:
            return faiss.read_index(VISUAL_INDEX_PATH)
        except Exception as e:
            print(f"[memory] Visual index load failed: {e}, rebuilding...")
            os.remove(VISUAL_INDEX_PATH)
    flat = faiss.IndexFlatIP(VISUAL_DIM)
    return faiss.IndexIDMap(flat)


def _save_visual_index(index):
    os.makedirs(os.path.dirname(VISUAL_INDEX_PATH), exist_ok=True)
    faiss.write_index(index, VISUAL_INDEX_PATH)


def _rebuild_visual_index():
    """Rebuild 512-dim CLIP visual index for all image memories."""
    from server.pipeline import get_clip_embedding, clip_model
    if clip_model is None:
        print("[memory] CLIP not loaded yet — visual index rebuild skipped")
        return _load_visual_index()

    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT id, image_path FROM memories WHERE image_path IS NOT NULL ORDER BY id"
    ).fetchall()
    conn.close()

    flat  = faiss.IndexFlatIP(VISUAL_DIM)
    index = faiss.IndexIDMap(flat)

    for memory_id, image_path in rows:
        if not image_path or not os.path.exists(image_path):
            continue
        try:
            vec = get_clip_embedding(image_path)
            index.add_with_ids(
                vec.reshape(1, -1),
                np.array([memory_id], dtype=np.int64),
            )
        except Exception as e:
            print(f"[memory] CLIP embed failed for id={memory_id}: {e}")

    _save_visual_index(index)
    print(f"[memory] visual index rebuilt: {index.ntotal} image memories at {VISUAL_DIM}-dim")
    return index


# ── SQLite schema ─────────────────────────────────────────────────────────────

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            tag TEXT DEFAULT '',
            created_at TEXT NOT NULL,
            access_count INTEGER DEFAULT 0,
            image_path TEXT DEFAULT NULL,
            image_description TEXT DEFAULT NULL
        )
    """)
    for col, typedef in (
        ("image_path",        "TEXT DEFAULT NULL"),
        ("image_description", "TEXT DEFAULT NULL"),
    ):
        try:
            conn.execute(f"ALTER TABLE memories ADD COLUMN {col} {typedef}")
        except Exception:
            pass
    conn.commit()
    conn.close()


# ── Save ──────────────────────────────────────────────────────────────────────

def save_memory(text, tag="", image_path=None, image_description=None, compress=False):
    """Insert a memory, update both FAISS indexes, return (stored_text, memory_id)."""
    global _save_counter

    settings    = load_settings()
    max_mem     = settings.get("max_memories", MAX_MEMORIES)
    auto_cleanup = settings.get("auto_cleanup", AUTO_CLEANUP)

    conn = sqlite3.connect(DB_PATH)
    count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    if count >= max_mem:
        if auto_cleanup:
            oldest = conn.execute("SELECT id FROM memories ORDER BY id ASC LIMIT 1").fetchone()
            if oldest:
                conn.execute("DELETE FROM memories WHERE id = ?", (oldest[0],))
                conn.commit()
                print(f"[memory] AUTO_CLEANUP: deleted oldest id={oldest[0]}")
        else:
            conn.close()
            print(f"[memory] MAX_MEMORIES ({max_mem}) reached, save aborted")
            return (None, None)
    conn.close()

    stored_text = text
    if image_description:
        stored_text = f"{text}\n[Visual: {image_description}]"

    if compress and len(stored_text.split()) > 20:
        try:
            from server.pipeline import compress_text
            stored_text = compress_text(stored_text)
        except Exception:
            pass

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute(
        "INSERT INTO memories (text, tag, created_at, access_count, image_path, image_description)"
        " VALUES (?, ?, ?, 0, ?, ?)",
        (stored_text, tag, datetime.now().isoformat(), image_path, image_description),
    )
    memory_id = cursor.lastrowid
    conn.commit()
    conn.close()

    # Text FAISS — InternVL3 text embedding for all memories (keyword fallback + semantic).
    text_index = _load_text_index()
    text_emb   = get_text_embedding(stored_text)
    text_index.add(text_emb.reshape(1, -1))
    _save_text_index(text_index)

    # Visual FAISS — CLIP embedding for image memories only, keyed by memory_id.
    if image_path and os.path.exists(image_path):
        try:
            from server.pipeline import get_clip_embedding, clip_model
            if clip_model is not None:
                vis_emb      = get_clip_embedding(image_path)
                visual_index = _load_visual_index()
                visual_index.add_with_ids(
                    vis_emb.reshape(1, -1),
                    np.array([memory_id], dtype=np.int64),
                )
                _save_visual_index(visual_index)
            else:
                print(f"[memory] CLIP not ready — visual index will be rebuilt after CLIP loads")
        except Exception as e:
            print(f"[memory] CLIP visual embed failed for id={memory_id}: {e}")

    _save_counter += 1
    if _save_counter % 10 == 0:
        deduplicate()

    return stored_text, memory_id


def update_memory_image(memory_id, image_description):
    """Append [Visual: ...] to an existing memory and rebuild the TEXT index.
    Called from background thread after InternVL3 describes the saved image.
    The visual FAISS index is unaffected — CLIP embedding is image-based, not text-based.
    """
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute("SELECT text FROM memories WHERE id = ?", (memory_id,)).fetchone()
    if not row:
        conn.close()
        return
    base_text = row[0]
    if "\n[Visual:" in base_text:
        base_text = base_text.split("\n[Visual:")[0]
    new_text = f"{base_text}\n[Visual: {image_description}]"
    conn.execute(
        "UPDATE memories SET text = ?, image_description = ? WHERE id = ?",
        (new_text, image_description, memory_id),
    )
    conn.commit()
    conn.close()
    try:
        _rebuild_text_index()
        print(f"[memory] text index updated after image description (id={memory_id})")
    except Exception as e:
        print(f"[memory] text index rebuild after image update failed: {e}")


# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve_memories(query, top_k=3, image_description=None):
    """Keyword-first retrieval from text FAISS; FAISS fallback for person queries."""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT id, text, image_description FROM memories ORDER BY id").fetchall()
    conn.close()

    if not rows:
        return []

    search_text = f"{query} {image_description}" if image_description else query
    is_person   = _is_person_query(query)

    kw_scored = []
    for row_id, text, img_desc in rows:
        score = _keyword_score(search_text, text)
        if is_person and img_desc:
            score = max(score, _keyword_score(search_text, img_desc))
        if score > 0:
            kw_scored.append((score, row_id, text))
    kw_scored.sort(reverse=True)

    KEYWORD_THRESH = 0.25
    results  = []
    seen_ids = set()
    conn     = sqlite3.connect(DB_PATH)
    for score, row_id, text in kw_scored[:top_k]:
        if score >= KEYWORD_THRESH:
            conn.execute("UPDATE memories SET access_count = access_count + 1 WHERE id = ?", (row_id,))
            results.append(text)
            seen_ids.add(row_id)

    # FAISS semantic fallback — person queries only (keyword-poor phrasing like "who is this?")
    if not results and is_person:
        text_index = _load_text_index()
        if text_index.ntotal > 0:
            q_vec        = get_embedding(search_text).reshape(1, -1)
            k            = min(top_k, text_index.ntotal)
            dists, idxs  = text_index.search(q_vec, k)
            # Positional mapping: text index rows follow ORDER BY id insertion order
            all_rows = conn.execute("SELECT id, text FROM memories ORDER BY id").fetchall()
            for dist, idx in zip(dists[0], idxs[0]):
                if idx < 0 or idx >= len(all_rows):
                    continue
                if dist >= PERSON_THRESHOLD:
                    row_id, text = all_rows[idx]
                    if row_id not in seen_ids:
                        conn.execute("UPDATE memories SET access_count = access_count + 1 WHERE id = ?", (row_id,))
                        results.append(text)
                        seen_ids.add(row_id)

    conn.commit()
    conn.close()
    return results


def retrieve_memories_by_image(clip_embedding, threshold=None):
    """
    Search the CLIP visual FAISS index for the closest image memory.

    clip_embedding: 512-dim float32 L2-normalised CLIP embedding.
    threshold:      minimum cosine similarity (default from settings, 0.72).
    Returns (memory_dict, score) or None.
    """
    if threshold is None:
        threshold = load_settings().get("visual_threshold", 0.72)

    visual_index = _load_visual_index()
    if visual_index.ntotal == 0:
        return None

    q_vec = clip_embedding.reshape(1, -1).astype(np.float32)
    k     = min(5, visual_index.ntotal)
    dists, ids = visual_index.search(q_vec, k)

    for dist, memory_id in zip(dists[0], ids[0]):
        if memory_id < 0:
            continue
        if float(dist) < threshold:
            continue
        # IDMap gives us the actual SQLite id — no positional mapping needed.
        conn = sqlite3.connect(DB_PATH)
        row  = conn.execute(
            "SELECT id, text, tag, created_at, access_count, image_path, image_description"
            " FROM memories WHERE id = ?",
            (int(memory_id),),
        ).fetchone()
        conn.close()
        if not row:
            continue
        row_id, text, tag, created_at, access_count, image_path, image_description = row
        return ({
            "id":                row_id,
            "text":              text,
            "tag":               tag,
            "created_at":        created_at,
            "access_count":      access_count,
            "image_path":        image_path,
            "image_description": image_description,
        }, float(dist))

    return None


# ── List / delete ─────────────────────────────────────────────────────────────

def list_memories():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT id, text, tag, created_at, access_count, image_path FROM memories ORDER BY id DESC"
    ).fetchall()
    conn.close()
    return [
        {
            "id":           r[0],
            "text":         r[1],
            "tag":          r[2],
            "created_at":   r[3],
            "access_count": r[4],
            "has_image":    r[5] is not None,
        }
        for r in rows
    ]


def delete_memory(memory_id):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
    conn.commit()
    conn.close()
    # Rebuild both indexes — text (positional) and visual (IDMap cleanup).
    try:
        _rebuild_text_index()
    except Exception as e:
        print(f"[memory] text index rebuild failed after delete (id={memory_id}): {e}")
    try:
        _rebuild_visual_index()
    except Exception as e:
        print(f"[memory] visual index rebuild failed after delete (id={memory_id}): {e}")


def delete_all_memories():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM memories")
    conn.commit()
    conn.close()
    for path in (TEXT_INDEX_PATH, VISUAL_INDEX_PATH):
        if os.path.exists(path):
            os.remove(path)
    print("[memory] all memories deleted, both indexes cleared")


# ── Deduplication ─────────────────────────────────────────────────────────────

def deduplicate():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT id, text FROM memories ORDER BY id").fetchall()
    conn.close()

    if len(rows) < 2:
        return

    embeddings = [get_embedding(text) for _, text in rows]
    to_delete  = set()

    for i in range(len(rows)):
        if rows[i][0] in to_delete:
            continue
        for j in range(i + 1, len(rows)):
            if rows[j][0] in to_delete:
                continue
            if float(np.dot(embeddings[i], embeddings[j])) > 0.95:
                to_delete.add(rows[j][0])

    if to_delete:
        conn = sqlite3.connect(DB_PATH)
        for del_id in to_delete:
            conn.execute("DELETE FROM memories WHERE id = ?", (del_id,))
        conn.commit()
        conn.close()
        try:
            _rebuild_text_index()
            _rebuild_visual_index()
        except Exception as e:
            print(f"[memory] index rebuild failed after dedup: {e}")
        print(f"[memory] deduplication removed {len(to_delete)} entries")
