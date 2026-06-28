package com.snapon.android.audio;

import java.io.File;
import java.io.IOException;

public interface OfflineSpeechSynthesizer {
    File synthesizeToWav(String text) throws IOException;
}
