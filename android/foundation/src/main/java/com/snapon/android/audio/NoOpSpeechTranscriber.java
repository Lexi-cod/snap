package com.snapon.android.audio;

import java.io.IOException;

public final class NoOpSpeechTranscriber implements SpeechTranscriber {
    @Override
    public String transcribe(AudioCaptureResult capture) throws IOException {
        throw new IOException("offline speech transcription engine is not installed");
    }
}
