package com.snapon.mobile

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import androidx.lifecycle.lifecycleScope
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat
import com.snapon.android.audio.AndroidPcmPushToTalkRecorder
import com.snapon.android.audio.SnapOnAudioService
import com.snapon.android.data.LocalMemoryRepository
import com.snapon.mobile.camera.CameraPreviewController
import com.snapon.mobile.runtime.AndroidSystemSpokenResponsePlayer
import com.snapon.mobile.runtime.PackagedSnapOnRuntime
import com.snapon.mobile.runtime.RuntimeSpeechTranscriber
import com.snapon.mobile.runtime.RuntimeTextEmbedder
import com.snapon.mobile.ui.SnapOnShellView
import com.snapon.mobile.workflow.SaveIntentParser
import com.snapon.mobile.workflow.SnapOnOrchestrator
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream

class MainActivity : ComponentActivity() {
    private lateinit var shellView: SnapOnShellView
    private lateinit var cameraPreviewController: CameraPreviewController
    private lateinit var memoryRepository: LocalMemoryRepository
    private lateinit var runtime: PackagedSnapOnRuntime
    private lateinit var audioService: SnapOnAudioService
    private lateinit var orchestrator: SnapOnOrchestrator

    private val permissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { grants ->
            val cameraGranted = grants[Manifest.permission.CAMERA] == true || hasPermission(Manifest.permission.CAMERA)
            val audioGranted = grants[Manifest.permission.RECORD_AUDIO] == true || hasPermission(Manifest.permission.RECORD_AUDIO)
            onPermissionsResolved(cameraGranted, audioGranted)
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        shellView = SnapOnShellView(this)
        setContentView(shellView.root)

        runtime = PackagedSnapOnRuntime(applicationContext)
        memoryRepository = LocalMemoryRepository(applicationContext, RuntimeTextEmbedder(runtime))
        audioService = SnapOnAudioService(
            AndroidPcmPushToTalkRecorder(File(cacheDir, "audio")),
            RuntimeSpeechTranscriber(runtime),
            AndroidSystemSpokenResponsePlayer(applicationContext)
        )
        orchestrator = SnapOnOrchestrator(memoryRepository, runtime)

        cameraPreviewController = CameraPreviewController(
            lifecycleOwner = this,
            previewView = shellView.previewView,
            onStatus = { status ->
                shellView.showStatus(status)
                if (status == CameraPreviewController.STATUS_ACTIVE) {
                    shellView.hideCameraPlaceholder()
                }
            }
        )

        shellView.onPushToTalkStart = ::beginPushToTalk
        shellView.onPushToTalkStop = ::finishPushToTalk
        shellView.onAsk = { question -> handleUserText(question) }
        shellView.onMemories = { showMemories() }

        requestRuntimePermissions()
        shellView.setMemoryCount(orchestrator.memoryCount())
        shellView.showAnswer("Point the camera, press the mic, and ask what to remember.")
        shellView.showStatus(runtime.runtimeStatusText())
    }

    private fun requestRuntimePermissions() {
        val missing = listOf(
            Manifest.permission.CAMERA,
            Manifest.permission.RECORD_AUDIO
        ).filterNot(::hasPermission)

        if (missing.isEmpty()) {
            onPermissionsResolved(cameraGranted = true, audioGranted = true)
        } else {
            permissionLauncher.launch(missing.toTypedArray())
        }
    }

    private fun onPermissionsResolved(cameraGranted: Boolean, audioGranted: Boolean) {
        if (cameraGranted) {
            cameraPreviewController.start()
        } else {
            shellView.showCameraPlaceholder("Camera permission is needed for visual questions.")
        }

        if (!audioGranted) {
            shellView.showStatus("Mic permission missing; typed questions still work.")
        } else {
            shellView.showStatus(runtime.runtimeStatusText())
        }
    }

    private fun hasPermission(permission: String): Boolean {
        return ContextCompat.checkSelfPermission(this, permission) == PackageManager.PERMISSION_GRANTED
    }

    private fun beginPushToTalk() {
        lifecycleScope.launch {
            runCatching {
                withContext(Dispatchers.IO) {
                    audioService.beginPushToTalk()
                }
            }.onSuccess {
                shellView.setListening(true)
                shellView.showStatus("Recording locally...")
            }.onFailure {
                shellView.setListening(false)
                shellView.showStatus("Microphone capture failed: ${it.message}")
            }
        }
    }

    private fun finishPushToTalk() {
        if (!audioService.isListening) {
            shellView.setListening(false)
            return
        }

        lifecycleScope.launch {
            shellView.showStatus("Transcribing on device...")
            val transcript = runCatching {
                withContext(Dispatchers.IO) {
                    audioService.finishPushToTalk()
                }
            }
            shellView.setListening(false)
            transcript.onSuccess { text ->
                shellView.setQuestion(text)
                handleUserText(text)
            }.onFailure {
                shellView.showStatus(it.message ?: "Speech transcription unavailable; use typed input.")
            }
        }
    }

    private fun handleUserText(question: String) {
        val cleanQuestion = question.trim()
        if (cleanQuestion.isEmpty()) {
            shellView.showStatus("Ask a question first")
            return
        }

        lifecycleScope.launch {
            shellView.showStatus("Running local query flow...")
            val imageBytes = cameraPreviewController.captureCurrentFrameJpeg()
            val shouldPersistFrame = SaveIntentParser.isSaveIntent(cleanQuestion)
            val imageUri = if (shouldPersistFrame) {
                withContext(Dispatchers.IO) {
                    persistFrame(imageBytes)
                }
            } else {
                null
            }
            val result = runCatching {
                withContext(Dispatchers.IO) {
                    orchestrator.handleTextInput(cleanQuestion, imageBytes, imageUri)
                }
            }

            result.onSuccess { flow ->
                shellView.showAnswer(flow.answer)
                shellView.setMemoryCount(flow.memoryCount)
                shellView.showStatus(flow.status)
                runCatching { audioService.speak(flow.answer) }
            }.onFailure {
                shellView.showStatus("Local flow failed: ${it.message}")
            }
        }
    }

    private fun showMemories() {
        shellView.showAnswer(orchestrator.memoryPreview())
        shellView.showStatus("Local memory preview")
    }

    private fun persistFrame(imageBytes: ByteArray?): String? {
        if (imageBytes == null) return null
        val imagesDir = File(filesDir, "memories/images")
        if (!imagesDir.exists()) {
            imagesDir.mkdirs()
        }
        val file = File(imagesDir, "memory-${System.currentTimeMillis()}.jpg")
        FileOutputStream(file).use { out ->
            out.write(imageBytes)
        }
        return file.absolutePath
    }

    override fun onDestroy() {
        super.onDestroy()
        runCatching { memoryRepository.close() }
        runCatching { audioService.close() }
    }
}
