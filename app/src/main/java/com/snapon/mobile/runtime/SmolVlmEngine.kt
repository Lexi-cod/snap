package com.snapon.mobile.runtime

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module
import org.pytorch.executorch.Tensor
import java.io.File
import java.io.FileOutputStream
import kotlin.math.min

/**
 * Runs SmolVLM-500M-Instruct locally via the ExecuTorch XNNPACK (CPU) backend,
 * matching the export contract in scripts/export_smolvlm_pte.py exactly:
 *   vision_encoder.pte  [1,3,512,512] f32  -> [1,64,960] f32
 *   tok_embedding.pte   [1,128] i64        -> [1,128,960] f32
 *   smolvlm_decoder.pte [1,128,960] f32 + [1,128] i64 position_ids -> [1,128,49280] f32 logits
 *
 * No KV cache: every generated token re-runs tok_embedding + decoder over the
 * full 128-token padded sequence (constant cost per step, not compounding,
 * since the export shape is fixed at 128 regardless of real content length).
 *
 * The chat-template scaffold below was verified byte-for-byte against the real
 * Idefics3Processor/AutoProcessor output for SmolVLM-500M-Instruct with
 * do_image_splitting=False (the single-tile case the vision encoder was
 * exported for): "<|im_start|>User:<fake_token_around_image><global-img>"
 * + "<image>"x64 + "<fake_token_around_image>{question}<end_of_utterance>\nAssistant:"
 */
class SmolVlmEngine(private val context: Context) {
    companion object {
        private const val IMAGE_SIZE = 512
        private const val HIDDEN_DIM = 960
        private const val N_VIS_TOKENS = 64
        private const val VOCAB_SIZE = 49280
        private const val MAX_SEQ_LEN = 128
        private const val MAX_NEW_TOKENS = 40
        private const val EOS_ID = 49279L
        private const val PAD_ID = 0L
        private const val IMAGE_TOKEN_START = 5 // index of the first <image> id in PREFIX_IDS

        private val PREFIX_IDS: LongArray =
            longArrayOf(1, 11126, 42, 49189, 49152) + LongArray(N_VIS_TOKENS) { 49190L } + longArrayOf(49189)
        private val SUFFIX_IDS = longArrayOf(EOS_ID, 198, 9519, 9531, 42)
    }

    private val modelsDir = File(context.filesDir, "smolvlm")

    private val tokenizer: BpeTokenizer by lazy {
        BpeTokenizer.fromAssets(context.assets, "tokenizers/smolvlm_500m_instruct/tokenizer.json")
    }

    private val visionEncoder: Module by lazy {
        Module.load(copyAssetToFile("xnnpack/vision_encoder.pte").absolutePath)
    }
    private val tokEmbedding: Module by lazy {
        Module.load(copyAssetToFile("xnnpack/tok_embedding.pte").absolutePath)
    }
    private val decoder: Module by lazy {
        Module.load(copyAssetToFile("xnnpack/smolvlm_decoder.pte").absolutePath)
    }

    /**
     * @param imageJpeg camera frame, or null if no frame was available (a neutral
     *   gray image is substituted so the prompt scaffold — which always reserves
     *   64 image-token slots — stays on a single, always-correct code path).
     */
    fun answer(question: String, imageJpeg: ByteArray?): String {
        val bitmap = imageJpeg
            ?.let { BitmapFactory.decodeByteArray(it, 0, it.size) }
            ?: Bitmap.createBitmap(IMAGE_SIZE, IMAGE_SIZE, Bitmap.Config.ARGB_8888).apply {
                eraseColor(Color.rgb(128, 128, 128))
            }
        val visionEmbeds = runVisionEncoder(preprocessImage(bitmap))

        val questionBudget = MAX_SEQ_LEN - PREFIX_IDS.size - SUFFIX_IDS.size
        val questionIds = tokenizer.encode(question).let { ids ->
            if (ids.size > questionBudget) ids.subList(0, questionBudget) else ids
        }

        val seq = ArrayList<Long>(MAX_SEQ_LEN).apply {
            addAll(PREFIX_IDS.toList())
            addAll(questionIds.map { it.toLong() })
            addAll(SUFFIX_IDS.toList())
        }

        val generated = mutableListOf<Int>()
        val positionIds = LongArray(MAX_SEQ_LEN) { it.toLong() }
        while (generated.size < MAX_NEW_TOKENS && seq.size < MAX_SEQ_LEN) {
            val embeds = embedSequence(seq, visionEmbeds)
            val logits = runDecoder(embeds, positionIds)
            val nextId = argmaxAt(logits, seq.size - 1)
            if (nextId.toLong() == EOS_ID) break
            generated.add(nextId)
            seq.add(nextId.toLong())
        }

        return tokenizer.decode(generated).trim()
    }

    /**
     * A lightweight semantic text embedding for memory save/retrieve: mean-pools
     * the token-embedding-table lookups for the text (no transformer layers, so
     * it's just an embedding-table read — much cheaper than a real decode pass)
     * and L2-normalizes the result. This only touches the already-loaded
     * [tokenizer]/[tokEmbedding], never [visionEncoder] or [decoder].
     */
    fun embed(text: String): FloatArray {
        val ids = tokenizer.encode(text)
        val realLen = min(ids.size, MAX_SEQ_LEN)
        val padded = LongArray(MAX_SEQ_LEN) { i -> if (i < realLen) ids[i].toLong() else PAD_ID }
        val idsTensor = Tensor.fromBlob(padded, longArrayOf(1, MAX_SEQ_LEN.toLong()))
        val embeds = tokEmbedding.forward(EValue.from(idsTensor))[0].toTensor().dataAsFloatArray

        val pooled = FloatArray(HIDDEN_DIM)
        for (pos in 0 until realLen) {
            val base = pos * HIDDEN_DIM
            for (d in 0 until HIDDEN_DIM) pooled[d] += embeds[base + d]
        }
        if (realLen > 0) {
            for (d in 0 until HIDDEN_DIM) pooled[d] = pooled[d] / realLen.toFloat()
        }

        var norm = 0f
        for (v in pooled) norm += v * v
        norm = kotlin.math.sqrt(norm)
        if (norm > 1e-6f) {
            for (d in pooled.indices) pooled[d] = pooled[d] / norm
        }
        return pooled
    }

    private fun copyAssetToFile(assetPath: String): File {
        val out = File(modelsDir, assetPath.substringAfterLast('/'))
        if (out.exists()) return out
        modelsDir.mkdirs()
        context.assets.open(assetPath).use { input ->
            FileOutputStream(out).use { output -> input.copyTo(output) }
        }
        return out
    }

    private fun preprocessImage(source: Bitmap): FloatArray {
        val resized = Bitmap.createScaledBitmap(source, IMAGE_SIZE, IMAGE_SIZE, true)
        val plane = IMAGE_SIZE * IMAGE_SIZE
        val pixels = IntArray(plane)
        resized.getPixels(pixels, 0, IMAGE_SIZE, 0, 0, IMAGE_SIZE, IMAGE_SIZE)

        // Matches processor_config.json: rescale 1/255 then normalize with
        // mean=std=0.5 per channel (-> range [-1, 1]), CHW layout.
        val out = FloatArray(3 * plane)
        for (i in 0 until plane) {
            val p = pixels[i]
            out[i] = (((p shr 16) and 0xFF) / 255f - 0.5f) / 0.5f
            out[plane + i] = (((p shr 8) and 0xFF) / 255f - 0.5f) / 0.5f
            out[2 * plane + i] = ((p and 0xFF) / 255f - 0.5f) / 0.5f
        }
        if (resized !== source) resized.recycle()
        return out
    }

    private fun runVisionEncoder(pixelValues: FloatArray): FloatArray {
        val tensor = Tensor.fromBlob(pixelValues, longArrayOf(1, 3, IMAGE_SIZE.toLong(), IMAGE_SIZE.toLong()))
        val out = visionEncoder.forward(EValue.from(tensor))
        return out[0].toTensor().dataAsFloatArray
    }

    private fun embedSequence(seq: List<Long>, visionEmbeds: FloatArray): FloatArray {
        val ids = LongArray(MAX_SEQ_LEN) { i -> if (i < seq.size) seq[i] else PAD_ID }
        val idsTensor = Tensor.fromBlob(ids, longArrayOf(1, MAX_SEQ_LEN.toLong()))
        val out = tokEmbedding.forward(EValue.from(idsTensor))
        val embeds = out[0].toTensor().dataAsFloatArray
        System.arraycopy(visionEmbeds, 0, embeds, IMAGE_TOKEN_START * HIDDEN_DIM, N_VIS_TOKENS * HIDDEN_DIM)
        return embeds
    }

    private fun runDecoder(embeds: FloatArray, positionIds: LongArray): FloatArray {
        val embedsTensor = Tensor.fromBlob(embeds, longArrayOf(1, MAX_SEQ_LEN.toLong(), HIDDEN_DIM.toLong()))
        val posTensor = Tensor.fromBlob(positionIds, longArrayOf(1, MAX_SEQ_LEN.toLong()))
        val out = decoder.forward(EValue.from(embedsTensor), EValue.from(posTensor))
        return out[0].toTensor().dataAsFloatArray
    }

    private fun argmaxAt(logits: FloatArray, position: Int): Int {
        val base = position * VOCAB_SIZE
        var bestIdx = 0
        var bestVal = Float.NEGATIVE_INFINITY
        for (v in 0 until VOCAB_SIZE) {
            val value = logits[base + v]
            if (value > bestVal) {
                bestVal = value
                bestIdx = v
            }
        }
        return bestIdx
    }
}
