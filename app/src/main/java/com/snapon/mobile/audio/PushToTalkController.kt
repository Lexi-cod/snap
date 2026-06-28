package com.snapon.mobile.audio

class PushToTalkController(
    private val onListeningChanged: (Boolean) -> Unit
) {
    private var listening = false

    fun start() {
        setListening(true)
    }

    fun stop() {
        setListening(false)
    }

    private fun setListening(next: Boolean) {
        if (listening == next) return
        listening = next
        onListeningChanged(next)
    }
}
