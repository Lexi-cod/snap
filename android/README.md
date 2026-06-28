# SnapOn Android Data and Audio Foundations

This directory contains the Android-native building blocks for the phone app.
It intentionally avoids the current Flask/browser runtime and keeps model
execution behind interfaces for the future ExecuTorch/QNN layer.

## Data layer

- `Memory`, `MemoryDraft`, and `SnapOnSettings` model the local records used by
  the Python prototype.
- `MemoryRepository` is the app-facing contract for saving, listing, deleting,
  retrieving, and updating local settings.
- `LocalMemoryRepository` persists memories and settings in an Android SQLite
  database through `SnapOnDatabaseHelper`.
- `MemorySearch` provides an offline keyword scorer so early Android builds can
  retrieve local memories before vector search is wired in.

## Audio layer

- `PushToTalkRecorder` captures short user utterances for offline STT.
- `SpeechTranscriber` is the future hook for Whisper/ExecuTorch transcription.
- `OfflineSpeechSynthesizer` is the future hook for Piper or another packaged
  offline TTS engine.
- `AndroidPcmPushToTalkRecorder` records microphone audio to a WAV file with
  `AudioRecord`.
- `AndroidSpokenResponsePlayer` plays synthesized WAV responses locally with
  `MediaPlayer`.
- `SnapOnAudioService` gives the app shell a simple push-to-talk facade while
  keeping STT and TTS engines replaceable.

The Android manifest declares microphone permission only. Camera, UI, and
ExecuTorch runtime wiring are intentionally left to their owning threads.
