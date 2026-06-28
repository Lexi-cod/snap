package com.snapon.android.audio;

import android.media.MediaPlayer;

import java.io.File;
import java.io.IOException;

public final class AndroidSpokenResponsePlayer implements SpokenResponsePlayer {
    private final OfflineSpeechSynthesizer synthesizer;
    private MediaPlayer player;

    public AndroidSpokenResponsePlayer(OfflineSpeechSynthesizer synthesizer) {
        if (synthesizer == null) {
            throw new IllegalArgumentException("synthesizer is required");
        }
        this.synthesizer = synthesizer;
    }

    @Override
    public synchronized void speak(String text) throws IOException {
        String trimmed = text == null ? "" : text.trim();
        if (trimmed.isEmpty()) {
            return;
        }

        stop();
        File wav = synthesizer.synthesizeToWav(trimmed);
        MediaPlayer next = new MediaPlayer();
        try {
            next.setDataSource(wav.getAbsolutePath());
            next.prepare();
            next.start();
            player = next;
        } catch (IOException | RuntimeException failure) {
            next.release();
            throw failure;
        }
    }

    @Override
    public synchronized void stop() {
        if (player == null) {
            return;
        }
        try {
            if (player.isPlaying()) {
                player.stop();
            }
        } catch (IllegalStateException ignored) {
            // Playback may have completed or failed before stop was requested.
        } finally {
            player.release();
            player = null;
        }
    }

    @Override
    public void close() {
        stop();
    }
}
