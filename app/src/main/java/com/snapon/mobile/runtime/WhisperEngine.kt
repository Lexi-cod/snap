package com.snapon.mobile.runtime

import android.content.Context
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module
import org.pytorch.executorch.Tensor
import java.io.File
import java.io.FileOutputStream
import kotlin.math.min

/**
 * Runs whisper-tiny.en locally via ExecuTorch XNNPACK (int8), using the
 * pre-exported "software-mansion/react-native-executorch" build of the model
 * (models/xnnpack/whisper_tiny_en.pte). That single .pte exposes two named
 * methods (not the default "forward", hence Module.execute(name, ...)):
 *   encode  audio[480000] f32 @16kHz mono      -> hidden[1,1500,384] f32
 *   decode  tokens[1,128] i64 + position_ids[128] i64 + hidden[1,1500,384] f32
 *           -> logits[1,128,51864] f32
 *
 * decode is the same fixed-length prefill/no-KV-cache pattern as
 * SmolVlmEngine's decoder: every generated token re-runs the full 128-slot
 * sequence (verified empirically — changing a token at position k only moves
 * logits at position >= k, confirming plain causal masking, no real cache).
 *
 * whisper-tiny.en has no language/task tokens (English-only checkpoint); the
 * real decoding prefix, taken from the model's HF generation_config, is just
 * <|startoftranscript|> (50257) followed by the forced <|notimestamps|>
 * (50362). Generation stops at <|endoftext|> (50256).
 */
class WhisperEngine(private val context: Context) {
    companion object {
        private const val AUDIO_SAMPLES = 480_000 // 30s @ 16kHz, fixed encoder input length
        private const val ENCODER_FRAMES = 1500
        private const val HIDDEN_DIM = 384
        private const val VOCAB_SIZE = 51864
        private const val MAX_SEQ_LEN = 128
        private const val MAX_NEW_TOKENS = 120
        private const val START_OF_TRANSCRIPT = 50257L
        private const val NO_TIMESTAMPS = 50362L
        private const val END_OF_TEXT = 50256L
        private const val PAD_ID = 0L
        private const val WAV_HEADER_BYTES = 44 // matches AndroidPcmPushToTalkRecorder's fixed canonical header
    }

    private val modelsDir = File(context.filesDir, "whisper")

    private val tokenizer: BpeTokenizer by lazy {
        BpeTokenizer.fromAssets(context.assets, "tokenizers/whisper_tiny_en/tokenizer.json")
    }

    private val whisper: Module by lazy {
        Module.load(copyAssetToFile("xnnpack/whisper_tiny_en.pte").absolutePath)
    }

    /** @param wavBytes a full RIFF/WAVE file: 44-byte canonical header + 16-bit PCM mono @16kHz. */
    fun transcribe(wavBytes: ByteArray): String {
        val samples = pcm16ToFloat(wavBytes)
        val hidden = runEncoder(samples)

        val seq = mutableListOf(START_OF_TRANSCRIPT, NO_TIMESTAMPS)
        val generated = mutableListOf<Int>()
        val positionIds = LongArray(MAX_SEQ_LEN) { it.toLong() }

        while (generated.size < MAX_NEW_TOKENS && seq.size < MAX_SEQ_LEN) {
            val logits = runDecoder(seq, hidden, positionIds)
            val nextId = argmaxAt(logits, seq.size - 1)
            if (nextId.toLong() == END_OF_TEXT) break
            generated.add(nextId)
            seq.add(nextId.toLong())
        }

        return tokenizer.decode(generated).trim()
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

    private fun pcm16ToFloat(wavBytes: ByteArray): FloatArray {
        val pcmStart = WAV_HEADER_BYTES
        val sampleCount = (wavBytes.size - pcmStart) / 2
        val out = FloatArray(AUDIO_SAMPLES)
        val n = min(sampleCount, AUDIO_SAMPLES)
        for (i in 0 until n) {
            val lo = wavBytes[pcmStart + i * 2].toInt() and 0xFF
            val hi = wavBytes[pcmStart + i * 2 + 1].toInt()
            out[i] = ((hi shl 8) or lo) / 32768f
        }
        return out // remaining samples beyond n stay zero-padded to AUDIO_SAMPLES
    }

    private fun runEncoder(samples: FloatArray): FloatArray {
        val tensor = Tensor.fromBlob(samples, longArrayOf(AUDIO_SAMPLES.toLong()))
        val out = whisper.execute("encode", EValue.from(tensor))
        return out[0].toTensor().dataAsFloatArray
    }

    private fun runDecoder(seq: List<Long>, encoderHidden: FloatArray, positionIds: LongArray): FloatArray {
        val ids = LongArray(MAX_SEQ_LEN) { i -> if (i < seq.size) seq[i] else PAD_ID }
        val idsTensor = Tensor.fromBlob(ids, longArrayOf(1, MAX_SEQ_LEN.toLong()))
        val posTensor = Tensor.fromBlob(positionIds, longArrayOf(MAX_SEQ_LEN.toLong()))
        val hiddenTensor = Tensor.fromBlob(encoderHidden, longArrayOf(1, ENCODER_FRAMES.toLong(), HIDDEN_DIM.toLong()))
        val out = whisper.execute("decode", EValue.from(idsTensor), EValue.from(posTensor), EValue.from(hiddenTensor))
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
