package com.snapon.runtime

enum class RuntimeBackend {
    EXECUTORCH_QNN,
    LOCAL_CPU,
    FALLBACK_TEXT_ONLY
}

data class RuntimeArtifactStatus(
    val id: String,
    val path: String,
    val required: Boolean,
    val present: Boolean,
    val sha256Ok: Boolean?
)

data class RuntimeReadiness(
    val ready: Boolean,
    val missingRequiredArtifacts: List<String>,
    val artifacts: List<RuntimeArtifactStatus>
)

data class VisualQuestionRequest(
    val question: String,
    val imageJpeg: ByteArray?,
    val memories: List<String> = emptyList()
)

data class VisualQuestionResponse(
    val answer: String,
    val backend: RuntimeBackend
)

interface SnapOnRuntime {
    fun readiness(): RuntimeReadiness

    suspend fun answerVisualQuestion(
        request: VisualQuestionRequest
    ): VisualQuestionResponse

    suspend fun transcribe(audioPcm16: ByteArray): String

    suspend fun synthesize(text: String): ByteArray?

    suspend fun embedText(text: String): FloatArray
}
