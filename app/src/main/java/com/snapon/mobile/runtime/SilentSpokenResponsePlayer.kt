package com.snapon.mobile.runtime

import com.snapon.android.audio.SpokenResponsePlayer
import java.io.IOException

class SilentSpokenResponsePlayer : SpokenResponsePlayer {
    override fun speak(text: String) {
        // Text output is the safe fallback until offline TTS assets are wired.
    }

    override fun stop() {
        // No-op.
    }

    override fun close() {
        // No-op.
    }
}
