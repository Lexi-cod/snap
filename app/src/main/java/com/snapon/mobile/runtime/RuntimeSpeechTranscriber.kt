package com.snapon.mobile.runtime

import com.snapon.android.audio.AudioCaptureResult
import com.snapon.android.audio.SpeechTranscriber
import java.io.IOException

class RuntimeSpeechTranscriber(
    private val runtime: PackagedSnapOnRuntime
) : SpeechTranscriber {
    override fun transcribe(capture: AudioCaptureResult): String {
        val bytes = capture.wavFile.readBytes()
        return kotlinx.coroutines.runBlocking {
            runtime.transcribe(bytes)
        }
    }
}
