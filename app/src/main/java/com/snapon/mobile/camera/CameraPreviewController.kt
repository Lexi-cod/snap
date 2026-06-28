package com.snapon.mobile.camera

import androidx.camera.core.CameraSelector
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner

class CameraPreviewController(
    private val lifecycleOwner: LifecycleOwner,
    private val previewView: PreviewView,
    private val onStatus: (String) -> Unit
) {
    companion object {
        const val STATUS_ACTIVE = "Camera preview active"
    }

    fun start() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(previewView.context)

        cameraProviderFuture.addListener(
            {
                runCatching {
                    val cameraProvider = cameraProviderFuture.get()
                    val preview = Preview.Builder().build().also { preview ->
                        preview.setSurfaceProvider(previewView.surfaceProvider)
                    }

                    cameraProvider.unbindAll()
                    cameraProvider.bindToLifecycle(
                        lifecycleOwner,
                        CameraSelector.DEFAULT_BACK_CAMERA,
                        preview
                    )
                    onStatus(STATUS_ACTIVE)
                }.onFailure {
                    onStatus("Camera preview unavailable; placeholder shell remains usable.")
                }
            },
            ContextCompat.getMainExecutor(previewView.context)
        )
    }
}
