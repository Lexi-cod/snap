package com.snapon.mobile.inference

import com.snapon.mobile.data.MemorySummary

data class VisualQueryInput(
    val question: String,
    val memoryContext: List<MemorySummary>
)

data class InferenceResult(
    val answer: String,
    val status: String
)

interface SnapOnInferenceEngine {
    fun answer(input: VisualQueryInput): InferenceResult
}

class PlaceholderSnapOnInferenceEngine : SnapOnInferenceEngine {
    override fun answer(input: VisualQueryInput): InferenceResult {
        val memoryHint = if (input.memoryContext.isEmpty()) {
            "No saved memories were needed for this placeholder response."
        } else {
            "I checked ${input.memoryContext.size} local memories before answering."
        }

        return InferenceResult(
            answer = "Camera, speech, memory, and ExecuTorch hooks are ready for on-device wiring.\n\nQuestion: ${input.question}\n\n$memoryHint",
            status = "Placeholder inference completed locally"
        )
    }
}
