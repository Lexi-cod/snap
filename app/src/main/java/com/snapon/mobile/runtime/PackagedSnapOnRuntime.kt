package com.snapon.mobile.runtime

import android.content.Context
import android.content.res.AssetManager
import com.snapon.runtime.RuntimeArtifactStatus
import com.snapon.runtime.RuntimeBackend
import com.snapon.runtime.RuntimeReadiness
import com.snapon.runtime.SnapOnRuntime
import com.snapon.runtime.VisualQuestionRequest
import com.snapon.runtime.VisualQuestionResponse
import org.json.JSONArray
import org.json.JSONObject
import java.io.IOException
import kotlin.math.min

class PackagedSnapOnRuntime(
    context: Context
) : SnapOnRuntime {
    private val assets: AssetManager = context.assets
    private val manifest by lazy { JSONObject(assets.open("artifact_manifest.json").bufferedReader().use { it.readText() }) }

    override fun readiness(): RuntimeReadiness {
        val artifacts = artifactStatuses()
        val missing = artifacts.filter { it.required && !it.present }.map { it.id }
        return RuntimeReadiness(
            ready = missing.isEmpty(),
            missingRequiredArtifacts = missing,
            artifacts = artifacts
        )
    }

    override suspend fun answerVisualQuestion(
        request: VisualQuestionRequest
    ): VisualQuestionResponse {
        val readiness = readiness()
        if (!readiness.ready) {
            return VisualQuestionResponse(
                answer = fallbackAnswer(request),
                backend = RuntimeBackend.FALLBACK_TEXT_ONLY
            )
        }

        return VisualQuestionResponse(
            answer = "Runtime artifacts are packaged, but the native ExecuTorch/QNN bridge is not linked yet.",
            backend = RuntimeBackend.LOCAL_CPU
        )
    }

    override suspend fun transcribe(audioPcm16: ByteArray): String {
        val readiness = readiness()
        if (!readiness.artifacts.any { it.id == "whisper_tiny_en_pte" && it.present }) {
            throw IOException("Speech-to-text artifact is not packaged yet. Use typed input for now.")
        }
        throw IOException("Speech runtime artifacts are packaged, but the native transcription bridge is not linked yet.")
    }

    override suspend fun synthesize(text: String): ByteArray? {
        return null
    }

    override suspend fun embedText(text: String): FloatArray {
        val tokens = text.lowercase().split(Regex("\\W+")).filter { it.isNotBlank() }
        val values = FloatArray(8)
        tokens.forEachIndexed { index, token ->
            values[index % values.size] += token.hashCode().toFloat()
        }
        return values
    }

    fun runtimeStatusText(): String {
        val readiness = readiness()
        return if (readiness.ready) {
            "Runtime artifacts packaged"
        } else {
            "Runtime missing: ${readiness.missingRequiredArtifacts.joinToString()}"
        }
    }

    private fun artifactStatuses(): List<RuntimeArtifactStatus> {
        val artifacts = manifest.getJSONArray("artifacts")
        return buildList {
            for (i in 0 until artifacts.length()) {
                val artifact = artifacts.getJSONObject(i)
                val path = artifact.getString("path")
                add(
                    RuntimeArtifactStatus(
                        id = artifact.getString("id"),
                        path = path,
                        required = artifact.optBoolean("required", false),
                        present = assetExists(path),
                        sha256Ok = null
                    )
                )
            }
        }
    }

    private fun assetExists(path: String): Boolean {
        val candidates = listOf(path, path.removePrefix("models/"), path.removePrefix("config/"))
            .distinct()

        return candidates.any { candidate ->
            try {
                assets.open(candidate).close()
                true
            } catch (_: IOException) {
                val listed = assets.list(candidate)
                listed != null && listed.isNotEmpty()
            }
        }
    }

    private fun fallbackAnswer(request: VisualQuestionRequest): String {
        val memoryLines = request.memories.map { it.trim() }.filter { it.isNotEmpty() }
        val top = memoryLines.firstOrNull()
        val question = request.question.trim()
        val visualHint = if (request.imageJpeg == null) {
            "No live camera frame was attached."
        } else {
            "A live frame was captured for future on-device VLM use."
        }

        return when {
            top != null && looksPersonalQuestion(question) ->
                "$top\n\n$visualHint"
            top != null ->
                "I found these local memories that may help:\n- ${
                    memoryLines.take(min(3, memoryLines.size)).joinToString("\n- ")
                }\n\n$visualHint"
            else ->
                "I do not have a matching saved memory yet. $visualHint"
        }
    }

    private fun looksPersonalQuestion(question: String): Boolean {
        val lower = question.lowercase()
        return listOf("who", "what did i save", "remember", "name", "this").any { it in lower }
    }
}
