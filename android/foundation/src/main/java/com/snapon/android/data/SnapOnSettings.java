package com.snapon.android.data;

public final class SnapOnSettings {
    public static final int DEFAULT_MAX_MEMORIES = 500;
    public static final boolean DEFAULT_AUTO_CLEANUP = true;
    public static final double DEFAULT_SIMILARITY_THRESHOLD = 0.30;
    public static final double DEFAULT_VISUAL_THRESHOLD = 0.72;

    private final int maxMemories;
    private final boolean autoCleanup;
    private final double similarityThreshold;
    private final double visualThreshold;

    public SnapOnSettings(
            int maxMemories,
            boolean autoCleanup,
            double similarityThreshold,
            double visualThreshold
    ) {
        this.maxMemories = Math.max(1, maxMemories);
        this.autoCleanup = autoCleanup;
        this.similarityThreshold = clamp01(similarityThreshold);
        this.visualThreshold = clamp01(visualThreshold);
    }

    public static SnapOnSettings defaults() {
        return new SnapOnSettings(
                DEFAULT_MAX_MEMORIES,
                DEFAULT_AUTO_CLEANUP,
                DEFAULT_SIMILARITY_THRESHOLD,
                DEFAULT_VISUAL_THRESHOLD
        );
    }

    public int getMaxMemories() {
        return maxMemories;
    }

    public boolean isAutoCleanup() {
        return autoCleanup;
    }

    public double getSimilarityThreshold() {
        return similarityThreshold;
    }

    public double getVisualThreshold() {
        return visualThreshold;
    }

    private static double clamp01(double value) {
        if (Double.isNaN(value)) {
            return 0.0;
        }
        return Math.max(0.0, Math.min(1.0, value));
    }
}
