package com.snapon.android.audio;

import java.io.Closeable;
import java.io.IOException;

public interface SpokenResponsePlayer extends Closeable {
    void speak(String text) throws IOException;

    void stop();
}
