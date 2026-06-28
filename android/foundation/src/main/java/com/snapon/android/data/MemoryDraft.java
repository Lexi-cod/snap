package com.snapon.android.data;

public final class MemoryDraft {
    private final String text;
    private final String tag;
    private final String imageUri;
    private final String imageDescription;

    public MemoryDraft(String text, String tag, String imageUri, String imageDescription) {
        this.text = Memory.requireText(text);
        this.tag = tag == null ? "" : tag.trim();
        this.imageUri = Memory.emptyToNull(imageUri);
        this.imageDescription = Memory.emptyToNull(imageDescription);
    }

    public static MemoryDraft textOnly(String text) {
        return new MemoryDraft(text, "", null, null);
    }

    public String getText() {
        return text;
    }

    public String getTag() {
        return tag;
    }

    public String getImageUri() {
        return imageUri;
    }

    public String getImageDescription() {
        return imageDescription;
    }
}
