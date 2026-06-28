package com.snapon.android.audio;

import java.io.IOException;

public interface SpeechTranscriber {
    String transcribe(AudioCaptureResult capture) throws IOException;
}
