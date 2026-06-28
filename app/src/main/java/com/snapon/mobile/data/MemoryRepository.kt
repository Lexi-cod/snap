package com.snapon.mobile.data

data class MemorySummary(
    val title: String,
    val detail: String,
    val tag: String
)

interface MemoryRepository {
    fun remember(memory: MemorySummary)
    fun recentMemories(limit: Int): List<MemorySummary>
    fun count(): Int
}

class InMemoryMemoryRepository : MemoryRepository {
    private val memories = mutableListOf<MemorySummary>()

    override fun remember(memory: MemorySummary) {
        memories.add(0, memory)
    }

    override fun recentMemories(limit: Int): List<MemorySummary> {
        return memories.take(limit)
    }

    override fun count(): Int {
        return memories.size
    }
}
