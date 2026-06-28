package com.snapon.android.audio;

import java.io.IOException;

public final class SnapOnAudioService {
    private final PushToTalkRecorder recorder;
    private final SpeechTranscriber transcriber;
    private final SpokenResponsePlayer spokenResponsePlayer;

    public SnapOnAudioService(
            PushToTalkRecorder recorder,
            SpeechTranscriber transcriber,
            SpokenResponsePlayer spokenResponsePlayer
    ) {
        if (recorder == null) {
            throw new IllegalArgumentException("recorder is required");
        }
        if (transcriber == null) {
            throw new IllegalArgumentException("transcriber is required");
        }
        if (spokenResponsePlayer == null) {
            throw new IllegalArgumentException("spokenResponsePlayer is required");
        }
        this.recorder = recorder;
        this.transcriber = transcriber;
        this.spokenResponsePlayer = spokenResponsePlayer;
    }

    public void beginPushToTalk() throws IOException {
        recorder.start();
    }

    public String finishPushToTalk() throws IOException {
        AudioCaptureResult capture = recorder.stop();
        return transcriber.transcribe(capture);
    }

    public void cancelPushToTalk() {
        recorder.cancel();
    }

    public boolean isListening() {
        return recorder.isRecording();
    }

    public void speak(String answer) throws IOException {
        spokenResponsePlayer.speak(answer);
    }

    public void stopSpeaking() {
        spokenResponsePlayer.stop();
    }
}
