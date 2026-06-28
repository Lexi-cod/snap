package com.snapon.android.audio;

import java.io.File;

public final class AudioCaptureResult {
    private final File wavFile;
    private final int sampleRateHz;
    private final int channelCount;
    private final long durationMillis;

    public AudioCaptureResult(File wavFile, int sampleRateHz, int channelCount, long durationMillis) {
        if (wavFile == null) {
            throw new IllegalArgumentException("wavFile is required");
        }
        this.wavFile = wavFile;
        this.sampleRateHz = sampleRateHz;
        this.channelCount = channelCount;
        this.durationMillis = Math.max(0L, durationMillis);
    }

    public File getWavFile() {
        return wavFile;
    }

    public int getSampleRateHz() {
        return sampleRateHz;
    }

    public int getChannelCount() {
        return channelCount;
    }

    public long getDurationMillis() {
        return durationMillis;
    }
}
