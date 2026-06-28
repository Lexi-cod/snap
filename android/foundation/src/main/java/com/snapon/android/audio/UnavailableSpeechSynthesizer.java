package com.snapon.android.audio;

import java.io.File;
import java.io.IOException;

public final class UnavailableSpeechSynthesizer implements OfflineSpeechSynthesizer {
    @Override
    public File synthesizeToWav(String text) throws IOException {
        throw new IOException("offline speech synthesis engine is not installed");
    }
}
