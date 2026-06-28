package com.snapon.android.audio;

import java.io.IOException;

public interface PushToTalkRecorder {
    void start() throws IOException;

    AudioCaptureResult stop() throws IOException;

    void cancel();

    boolean isRecording();
}
