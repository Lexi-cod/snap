package com.snapon.android.audio;

import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.concurrent.atomic.AtomicBoolean;

public final class AndroidPcmPushToTalkRecorder implements PushToTalkRecorder {
    private static final int SAMPLE_RATE_HZ = 16_000;
    private static final int CHANNEL_CONFIG = AudioFormat.CHANNEL_IN_MONO;
    private static final int AUDIO_FORMAT = AudioFormat.ENCODING_PCM_16BIT;
    private static final int CHANNEL_COUNT = 1;
    private static final int BITS_PER_SAMPLE = 16;

    private final File cacheDirectory;
    private final AtomicBoolean recording = new AtomicBoolean(false);

    private AudioRecord audioRecord;
    private ByteArrayOutputStream pcmBuffer;
    private Thread captureThread;
    private long startedAtMillis;

    public AndroidPcmPushToTalkRecorder(File cacheDirectory) {
        if (cacheDirectory == null) {
            throw new IllegalArgumentException("cacheDirectory is required");
        }
        this.cacheDirectory = cacheDirectory;
    }

    @Override
    public synchronized void start() throws IOException {
        if (recording.get()) {
            return;
        }
        if (!cacheDirectory.exists() && !cacheDirectory.mkdirs()) {
            throw new IOException("unable to create audio cache directory: " + cacheDirectory);
        }

        int minBuffer = AudioRecord.getMinBufferSize(SAMPLE_RATE_HZ, CHANNEL_CONFIG, AUDIO_FORMAT);
        if (minBuffer <= 0) {
            throw new IOException("microphone buffer size unavailable");
        }
        int bufferSize = minBuffer * 2;
        AudioRecord nextRecord = new AudioRecord(
                MediaRecorder.AudioSource.VOICE_RECOGNITION,
                SAMPLE_RATE_HZ,
                CHANNEL_CONFIG,
                AUDIO_FORMAT,
                bufferSize
        );
        if (nextRecord.getState() != AudioRecord.STATE_INITIALIZED) {
            nextRecord.release();
            throw new IOException("microphone failed to initialize");
        }

        pcmBuffer = new ByteArrayOutputStream();
        audioRecord = nextRecord;
        startedAtMillis = System.currentTimeMillis();
        recording.set(true);
        audioRecord.startRecording();
        captureThread = new Thread(new CaptureLoop(nextRecord, bufferSize), "snapon-ptt-recorder");
        captureThread.start();
    }

    @Override
    public synchronized AudioCaptureResult stop() throws IOException {
        if (!recording.get()) {
            throw new IOException("push-to-talk recorder is not active");
        }
        recording.set(false);
        AudioRecord stopped = audioRecord;
        audioRecord = null;
        if (stopped != null) {
            try {
                stopped.stop();
            } catch (IllegalStateException ignored) {
                // The recorder may already be stopped when permissions are revoked.
            }
        }
        joinCaptureThread();
        if (stopped != null) {
            stopped.release();
        }

        byte[] pcm = pcmBuffer == null ? new byte[0] : pcmBuffer.toByteArray();
        pcmBuffer = null;
        long durationMillis = System.currentTimeMillis() - startedAtMillis;
        File wav = File.createTempFile("snapon-ptt-", ".wav", cacheDirectory);
        writeWav(wav, pcm);
        return new AudioCaptureResult(wav, SAMPLE_RATE_HZ, CHANNEL_COUNT, durationMillis);
    }

    @Override
    public synchronized void cancel() {
        if (!recording.getAndSet(false)) {
            return;
        }
        if (audioRecord != null) {
            try {
                audioRecord.stop();
            } catch (IllegalStateException ignored) {
                // Best-effort cleanup.
            }
        }
        joinCaptureThread();
        if (audioRecord != null) {
            audioRecord.release();
            audioRecord = null;
        }
        pcmBuffer = null;
    }

    @Override
    public boolean isRecording() {
        return recording.get();
    }

    private void joinCaptureThread() {
        Thread thread = captureThread;
        captureThread = null;
        if (thread == null) {
            return;
        }
        try {
            thread.join(1_000L);
        } catch (InterruptedException interrupted) {
            Thread.currentThread().interrupt();
        }
    }

    private void appendPcm(byte[] data, int length) {
        synchronized (this) {
            if (pcmBuffer != null && length > 0) {
                pcmBuffer.write(data, 0, length);
            }
        }
    }

    private static void writeWav(File file, byte[] pcm) throws IOException {
        int byteRate = SAMPLE_RATE_HZ * CHANNEL_COUNT * BITS_PER_SAMPLE / 8;
        try (FileOutputStream out = new FileOutputStream(file)) {
            out.write(ascii("RIFF"));
            writeIntLe(out, 36 + pcm.length);
            out.write(ascii("WAVE"));
            out.write(ascii("fmt "));
            writeIntLe(out, 16);
            writeShortLe(out, (short) 1);
            writeShortLe(out, (short) CHANNEL_COUNT);
            writeIntLe(out, SAMPLE_RATE_HZ);
            writeIntLe(out, byteRate);
            writeShortLe(out, (short) (CHANNEL_COUNT * BITS_PER_SAMPLE / 8));
            writeShortLe(out, (short) BITS_PER_SAMPLE);
            out.write(ascii("data"));
            writeIntLe(out, pcm.length);
            out.write(pcm);
        }
    }

    private static byte[] ascii(String value) {
        return value.getBytes(java.nio.charset.StandardCharsets.US_ASCII);
    }

    private static void writeIntLe(FileOutputStream out, int value) throws IOException {
        out.write(ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putInt(value).array());
    }

    private static void writeShortLe(FileOutputStream out, short value) throws IOException {
        out.write(ByteBuffer.allocate(2).order(ByteOrder.LITTLE_ENDIAN).putShort(value).array());
    }

    private final class CaptureLoop implements Runnable {
        private final AudioRecord record;
        private final int bufferSize;

        CaptureLoop(AudioRecord record, int bufferSize) {
            this.record = record;
            this.bufferSize = bufferSize;
        }

        @Override
        public void run() {
            byte[] buffer = new byte[bufferSize];
            while (recording.get()) {
                int read = record.read(buffer, 0, buffer.length);
                if (read > 0) {
                    appendPcm(buffer, read);
                }
            }
        }
    }
}
