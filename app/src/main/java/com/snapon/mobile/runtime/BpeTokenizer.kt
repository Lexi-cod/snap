package com.snapon.mobile.runtime

import android.content.res.AssetManager
import org.json.JSONObject

/**
 * Byte-level BPE tokenizer (GPT2Tokenizer family) matching SmolVLM-500M-Instruct's
 * tokenizer.json (model.type == "BPE", pre_tokenizer.type == "ByteLevel").
 *
 * Only implements encode/decode of plain text. The fixed chat-template scaffold
 * (<|im_start|>, image placeholder block, "Assistant:", etc.) is not tokenized
 * generically here — SmolVlmEngine hardcodes those ids directly since they never
 * change, verified against the real HF processor output. This class only needs to
 * turn free-form question text into ids and generated ids back into text.
 */
class BpeTokenizer private constructor(
    private val vocab: Map<String, Int>,
    private val idToToken: Map<Int, String>,
    private val mergeRanks: Map<Pair<String, String>, Int>,
    private val specialIds: Set<Int>
) {
    companion object {
        private val PRE_TOKENIZE_REGEX = Regex(
            "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+"
        )

        fun fromAssets(assets: AssetManager, tokenizerJsonPath: String): BpeTokenizer {
            val json = assets.open(tokenizerJsonPath).bufferedReader().use { it.readText() }
            return fromJson(json)
        }

        fun fromJson(json: String): BpeTokenizer {
            val root = JSONObject(json)
            val model = root.getJSONObject("model")

            val vocabJson = model.getJSONObject("vocab")
            val vocab = HashMap<String, Int>(vocabJson.length() * 2)
            val keys = vocabJson.keys()
            while (keys.hasNext()) {
                val token = keys.next()
                vocab[token] = vocabJson.getInt(token)
            }

            val mergesJson = model.getJSONArray("merges")
            val mergeRanks = HashMap<Pair<String, String>, Int>(mergesJson.length() * 2)
            for (i in 0 until mergesJson.length()) {
                val pair = mergesJson.getJSONArray(i)
                mergeRanks[Pair(pair.getString(0), pair.getString(1))] = i
            }

            val specialIds = HashSet<Int>()
            if (root.has("added_tokens")) {
                val added = root.getJSONArray("added_tokens")
                for (i in 0 until added.length()) {
                    val entry = added.getJSONObject(i)
                    if (entry.optBoolean("special", false)) {
                        specialIds.add(entry.getInt("id"))
                    }
                    // added tokens (special or not) must also be addressable by content
                    // for completeness, though encode() here never needs to produce them.
                    vocab[entry.getString("content")] = entry.getInt("id")
                }
            }

            val idToToken = HashMap<Int, String>(vocab.size * 2)
            for ((token, id) in vocab) {
                idToToken[id] = token
            }

            return BpeTokenizer(vocab, idToToken, mergeRanks, specialIds)
        }

        private val BYTE_TO_UNICODE: Map<Int, Char> = buildByteToUnicode()
        private val UNICODE_TO_BYTE: Map<Char, Int> =
            BYTE_TO_UNICODE.entries.associate { (b, c) -> c to b }

        private fun buildByteToUnicode(): Map<Int, Char> {
            val bs = mutableListOf<Int>()
            bs.addAll('!'.code..'~'.code)
            bs.addAll(0xA1..0xAC)
            bs.addAll(0xAE..0xFF)
            val cs = bs.toMutableList()
            var n = 0
            for (b in 0..255) {
                if (b !in bs) {
                    bs.add(b)
                    cs.add(256 + n)
                    n++
                }
            }
            val map = HashMap<Int, Char>(bs.size * 2)
            for (i in bs.indices) {
                map[bs[i]] = cs[i].toChar()
            }
            return map
        }
    }

    private val bpeCache = HashMap<String, List<String>>()

    fun encode(text: String): List<Int> {
        val ids = mutableListOf<Int>()
        for (match in PRE_TOKENIZE_REGEX.findAll(text)) {
            val piece = match.value
            val byteMapped = buildString {
                for (b in piece.toByteArray(Charsets.UTF_8)) {
                    append(BYTE_TO_UNICODE.getValue(b.toInt() and 0xFF))
                }
            }
            for (sub in bpe(byteMapped)) {
                val id = vocab[sub]
                if (id != null) {
                    ids.add(id)
                } else {
                    // Should not happen for byte-level BPE with full single-byte
                    // vocab coverage, but fall back to per-character lookup rather
                    // than dropping the piece silently.
                    for (ch in sub) {
                        vocab[ch.toString()]?.let { ids.add(it) }
                    }
                }
            }
        }
        return ids
    }

    fun decode(ids: List<Int>): String {
        val bytes = mutableListOf<Byte>()
        for (id in ids) {
            if (id in specialIds) continue
            val token = idToToken[id] ?: continue
            for (ch in token) {
                UNICODE_TO_BYTE[ch]?.let { bytes.add(it.toByte()) }
            }
        }
        return String(bytes.toByteArray(), Charsets.UTF_8)
    }

    private fun bpe(token: String): List<String> {
        if (token.length <= 1) return listOf(token)
        bpeCache[token]?.let { return it }

        var word = token.map { it.toString() }
        while (word.size > 1) {
            var bestRank = Int.MAX_VALUE
            var bestIdx = -1
            for (i in 0 until word.size - 1) {
                val rank = mergeRanks[Pair(word[i], word[i + 1])] ?: continue
                if (rank < bestRank) {
                    bestRank = rank
                    bestIdx = i
                }
            }
            if (bestIdx == -1) break
            val merged = word[bestIdx] + word[bestIdx + 1]
            word = word.subList(0, bestIdx) + listOf(merged) + word.subList(bestIdx + 2, word.size)
        }
        bpeCache[token] = word
        return word
    }
}
