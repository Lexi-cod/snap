package com.snapon.android.data;

import java.util.Objects;

public final class Memory {
    private final long id;
    private final String text;
    private final String tag;
    private final long createdAtEpochMillis;
    private final int accessCount;
    private final String imageUri;
    private final String imageDescription;

    public Memory(
            long id,
            String text,
            String tag,
            long createdAtEpochMillis,
            int accessCount,
            String imageUri,
            String imageDescription
    ) {
        this.id = id;
        this.text = requireText(text);
        this.tag = tag == null ? "" : tag;
        this.createdAtEpochMillis = createdAtEpochMillis;
        this.accessCount = Math.max(0, accessCount);
        this.imageUri = emptyToNull(imageUri);
        this.imageDescription = emptyToNull(imageDescription);
    }

    public long getId() {
        return id;
    }

    public String getText() {
        return text;
    }

    public String getTag() {
        return tag;
    }

    public long getCreatedAtEpochMillis() {
        return createdAtEpochMillis;
    }

    public int getAccessCount() {
        return accessCount;
    }

    public String getImageUri() {
        return imageUri;
    }

    public String getImageDescription() {
        return imageDescription;
    }

    public boolean hasImage() {
        return imageUri != null;
    }

    public String displayText() {
        int visualMarker = text.indexOf("\n[Visual:");
        return visualMarker >= 0 ? text.substring(0, visualMarker).trim() : text;
    }

    static String requireText(String text) {
        String trimmed = text == null ? "" : text.trim();
        if (trimmed.isEmpty()) {
            throw new IllegalArgumentException("memory text is required");
        }
        return trimmed;
    }

    static String emptyToNull(String value) {
        if (value == null) {
            return null;
        }
        String trimmed = value.trim();
        return trimmed.isEmpty() ? null : trimmed;
    }

    @Override
    public boolean equals(Object other) {
        if (this == other) {
            return true;
        }
        if (!(other instanceof Memory)) {
            return false;
        }
        Memory memory = (Memory) other;
        return id == memory.id
                && createdAtEpochMillis == memory.createdAtEpochMillis
                && accessCount == memory.accessCount
                && text.equals(memory.text)
                && tag.equals(memory.tag)
                && Objects.equals(imageUri, memory.imageUri)
                && Objects.equals(imageDescription, memory.imageDescription);
    }

    @Override
    public int hashCode() {
        return Objects.hash(id, text, tag, createdAtEpochMillis, accessCount, imageUri, imageDescription);
    }
}
