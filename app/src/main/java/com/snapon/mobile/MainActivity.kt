package com.snapon.mobile

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat
import com.snapon.mobile.audio.PushToTalkController
import com.snapon.mobile.camera.CameraPreviewController
import com.snapon.mobile.data.InMemoryMemoryRepository
import com.snapon.mobile.data.MemorySummary
import com.snapon.mobile.inference.PlaceholderSnapOnInferenceEngine
import com.snapon.mobile.inference.VisualQueryInput
import com.snapon.mobile.ui.SnapOnShellView

class MainActivity : ComponentActivity() {
    private lateinit var shellView: SnapOnShellView
    private lateinit var cameraPreviewController: CameraPreviewController
    private lateinit var pushToTalkController: PushToTalkController

    private val memories = InMemoryMemoryRepository()
    private val inferenceEngine = PlaceholderSnapOnInferenceEngine()

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
        pushToTalkController = PushToTalkController(
            onListeningChanged = { isListening ->
                shellView.setListening(isListening)
                shellView.showStatus(if (isListening) "Listening locally..." else "Ready")
            }
        )

        shellView.onPushToTalkStart = pushToTalkController::start
        shellView.onPushToTalkStop = pushToTalkController::stop
        shellView.onAsk = { question -> answerQuestion(question) }
        shellView.onMemories = { showMemories() }

        requestRuntimePermissions()
        shellView.showAnswer("Point the camera, press the mic, and ask what to remember.")
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
            shellView.showStatus("On-device shell ready")
        }
    }

    private fun hasPermission(permission: String): Boolean {
        return ContextCompat.checkSelfPermission(this, permission) == PackageManager.PERMISSION_GRANTED
    }

    private fun answerQuestion(question: String) {
        val cleanQuestion = question.trim()
        if (cleanQuestion.isEmpty()) {
            shellView.showStatus("Ask a question first")
            return
        }

        shellView.showStatus("Running local placeholder inference")
        val result = inferenceEngine.answer(
            VisualQueryInput(
                question = cleanQuestion,
                memoryContext = memories.recentMemories(limit = 4)
            )
        )

        shellView.showAnswer(result.answer)
        memories.remember(
            MemorySummary(
                title = "Visual question",
                detail = cleanQuestion,
                tag = "demo"
            )
        )
        shellView.setMemoryCount(memories.count())
        shellView.showStatus(result.status)
    }

    private fun showMemories() {
        val recent = memories.recentMemories(limit = 3)
        val text = if (recent.isEmpty()) {
            "No memories saved yet."
        } else {
            recent.joinToString(separator = "\n\n") { memory ->
                "${memory.title}\n${memory.detail}"
            }
        }

        shellView.showAnswer(text)
        shellView.showStatus("Local memory preview")
    }
}
