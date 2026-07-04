package com.snapon.mobile.runtime

import android.content.Context
import android.speech.tts.TextToSpeech
import com.snapon.android.audio.SpokenResponsePlayer
import java.util.Locale
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Speaks answers using the device's built-in TextToSpeech engine rather than
 * a bundled neural voice model. Real Piper (named in the README) needs
 * espeak-ng for text-to-phoneme conversion, which means cross-compiling and
 * JNI-bridging a native C library for Android — out of scope for now. This
 * is still fully on-device/offline once the system voice data is installed
 * (the Android norm — no network call here), just not a model this project
 * trained or exported itself.
 */
class AndroidSystemSpokenResponsePlayer(context: Context) : SpokenResponsePlayer {
    private val ready = AtomicBoolean(false)
    private var tts: TextToSpeech? = null

    init {
        tts = TextToSpeech(context.applicationContext) { status ->
            if (status == TextToSpeech.SUCCESS) {
                tts?.language = Locale.US
                ready.set(true)
            }
        }
    }

    override fun speak(text: String) {
        val trimmed = text.trim()
        if (trimmed.isEmpty() || !ready.get()) {
            return
        }
        tts?.speak(trimmed, TextToSpeech.QUEUE_FLUSH, null, "snapon-answer")
    }

    override fun stop() {
        tts?.stop()
    }

    override fun close() {
        tts?.stop()
        tts?.shutdown()
        tts = null
    }
}
