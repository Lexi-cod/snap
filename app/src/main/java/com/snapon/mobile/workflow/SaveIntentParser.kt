package com.snapon.mobile.workflow

object SaveIntentParser {
    private val saveVerbs = setOf("save", "remember", "store", "memorize", "note", "keep")
    private val savePrefixes = listOf("save ", "remember ", "store ", "memorize ", "note that ", "keep this ")
    private val savePhrases = listOf("add to memory", "don't forget", "take note")

    fun isSaveIntent(text: String): Boolean {
        val lower = text.trim().lowercase()
        return savePrefixes.any(lower::startsWith) || savePhrases.any(lower::contains)
    }

    fun extractContent(text: String): String {
        val stripped = text.trim()
        val firstSpace = stripped.indexOf(' ')
        if (firstSpace != -1) {
            val verb = stripped.substring(0, firstSpace).lowercase().trim(',', '.', '!')
            if (verb in saveVerbs) {
                return stripped.substring(firstSpace + 1).trim()
            }
        }

        val lower = stripped.lowercase()
        for (phrase in savePhrases) {
            if (phrase in lower) {
                val start = lower.indexOf(phrase) + phrase.length
                return stripped.substring(start).trim().trimStart(':').trim().ifEmpty { stripped }
            }
        }
        return stripped
    }
}
