package com.snapon.android.data;

public interface TextEmbedder {
    /** Returns an L2-normalized embedding, or an empty array if embedding is unavailable. */
    float[] embed(String text);
}
