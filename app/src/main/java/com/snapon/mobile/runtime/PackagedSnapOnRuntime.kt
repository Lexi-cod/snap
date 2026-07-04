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
    private val context: Context
) : SnapOnRuntime {
    private val assets: AssetManager = context.assets
    private val manifest by lazy { JSONObject(assets.open("artifact_manifest.json").bufferedReader().use { it.readText() }) }

    // Lazy: only loads the .pte files into ExecuTorch Modules on first real
    // use, not at PackagedSnapOnRuntime construction time.
    private val engine: SmolVlmEngine by lazy { SmolVlmEngine(context) }
    private val whisperEngine: WhisperEngine by lazy { WhisperEngine(context) }

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

        return try {
            val answer = engine.answer(request.question, request.imageJpeg)
            VisualQuestionResponse(
                answer = answer.ifBlank { fallbackAnswer(request) },
                backend = RuntimeBackend.LOCAL_CPU
            )
        } catch (e: Exception) {
            VisualQuestionResponse(
                answer = "On-device model failed: ${e.message}\n\n${fallbackAnswer(request)}",
                backend = RuntimeBackend.FALLBACK_TEXT_ONLY
            )
        }
    }

    override suspend fun transcribe(audioPcm16: ByteArray): String {
        val readiness = readiness()
        if (!readiness.artifacts.any { it.id == "whisper_tiny_en_pte" && it.present }) {
            throw IOException("Speech-to-text artifact is not packaged yet. Use typed input for now.")
        }
        val text = whisperEngine.transcribe(audioPcm16)
        if (text.isBlank()) {
            throw IOException("Could not transcribe that clip; try again or use typed input.")
        }
        return text
    }

    override suspend fun synthesize(text: String): ByteArray? {
        return null
    }

    override suspend fun embedText(text: String): FloatArray {
        if (text.isBlank()) return FloatArray(0)
        return try {
            engine.embed(text)
        } catch (e: Exception) {
            FloatArray(0)
        }
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
