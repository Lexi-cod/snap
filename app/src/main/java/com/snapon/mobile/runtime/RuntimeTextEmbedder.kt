package com.snapon.mobile.runtime

import com.snapon.android.data.TextEmbedder
import com.snapon.runtime.SnapOnRuntime
import kotlinx.coroutines.runBlocking

class RuntimeTextEmbedder(private val runtime: SnapOnRuntime) : TextEmbedder {
    override fun embed(text: String): FloatArray = runBlocking { runtime.embedText(text) }
}
