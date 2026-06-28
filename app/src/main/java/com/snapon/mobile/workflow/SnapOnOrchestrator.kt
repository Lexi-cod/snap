package com.snapon.mobile.workflow

import com.snapon.android.data.Memory
import com.snapon.android.data.MemoryDraft
import com.snapon.android.data.MemoryRepository
import com.snapon.runtime.RuntimeBackend
import com.snapon.runtime.SnapOnRuntime
import com.snapon.runtime.VisualQuestionRequest
import java.io.IOException

data class FlowResult(
    val answer: String,
    val status: String,
    val memoryCount: Int,
    val backend: RuntimeBackend? = null
)

class SnapOnOrchestrator(
    private val memoryRepository: MemoryRepository,
    private val runtime: SnapOnRuntime
) {
    suspend fun handleTextInput(
        rawText: String,
        imageBytes: ByteArray?,
        imageUri: String?
    ): FlowResult {
        val text = rawText.trim()
        require(text.isNotEmpty()) { "Input text is required" }

        return if (SaveIntentParser.isSaveIntent(text)) {
            val content = SaveIntentParser.extractContent(text)
            val memory = memoryRepository.save(
                MemoryDraft(
                    content,
                    "saved",
                    imageUri,
                    null
                )
            )
            FlowResult(
                answer = "Saved locally:\n${memory.displayText()}",
                status = "Memory saved on device",
                memoryCount = memoryRepository.list().size
            )
        } else {
            val retrieved = memoryRepository.retrieve(text, 4)
            val response = runtime.answerVisualQuestion(
                VisualQuestionRequest(
                    question = text,
                    imageJpeg = imageBytes,
                    memories = retrieved.map(Memory::displayText)
                )
            )
            FlowResult(
                answer = response.answer,
                status = statusFor(response.backend, retrieved.size),
                memoryCount = memoryRepository.list().size,
                backend = response.backend
            )
        }
    }

    fun memoryPreview(limit: Int = 6): String {
        val memories = memoryRepository.list().take(limit)
        if (memories.isEmpty()) {
            return "No memories saved yet."
        }
        return memories.joinToString(separator = "\n\n") { memory ->
            buildString {
                append(memory.displayText())
                if (memory.tag.isNotBlank()) {
                    append("\nTag: ")
                    append(memory.tag)
                }
            }
        }
    }

    fun memoryCount(): Int = try {
        memoryRepository.list().size
    } catch (_: IOException) {
        0
    }

    private fun statusFor(backend: RuntimeBackend, retrievedCount: Int): String {
        val source = if (retrievedCount > 0) "using $retrievedCount local memories" else "with no memory match"
        return when (backend) {
            RuntimeBackend.EXECUTORCH_QNN -> "Answered on Snapdragon QNN ($source)"
            RuntimeBackend.LOCAL_CPU -> "Answered with local packaged runtime ($source)"
            RuntimeBackend.FALLBACK_TEXT_ONLY -> "Answered with local fallback ($source)"
        }
    }
}
