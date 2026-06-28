package com.snapon.mobile.ui

import android.content.Context
import android.graphics.Color
import android.graphics.Typeface
import android.view.MotionEvent
import android.view.Gravity
import android.view.View
import android.widget.Button
import android.widget.EditText
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import androidx.camera.view.PreviewView
import com.snapon.mobile.R

class SnapOnShellView(context: Context) {
    val root: View
    val previewView: PreviewView

    var onPushToTalkStart: () -> Unit = {}
    var onPushToTalkStop: () -> Unit = {}
    var onAsk: (String) -> Unit = {}
    var onMemories: () -> Unit = {}

    private val statusText: TextView
    private val answerText: TextView
    private val questionInput: EditText
    private val pushToTalkButton: Button
    private val memoryButton: Button
    private val cameraPlaceholder: TextView

    init {
        val outer = LinearLayout(context).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(context.color(R.color.snapon_surface))
            setPadding(24, 24, 24, 18)
        }

        val title = TextView(context).apply {
            text = "SnapOn"
            textSize = 30f
            typeface = Typeface.DEFAULT_BOLD
            setTextColor(context.color(R.color.snapon_ink))
        }
        outer.addView(title)

        statusText = TextView(context).apply {
            text = "Starting local shell"
            textSize = 14f
            setTextColor(context.color(R.color.snapon_panel))
        }
        outer.addView(statusText)

        val cameraFrame = FrameLayout(context).apply {
            setBackgroundColor(Color.BLACK)
        }
        previewView = PreviewView(context).apply {
            scaleType = PreviewView.ScaleType.FILL_CENTER
        }
        cameraPlaceholder = TextView(context).apply {
            text = "Camera preview"
            gravity = Gravity.CENTER
            textSize = 18f
            setTextColor(Color.WHITE)
        }
        cameraFrame.addView(
            previewView,
            FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                FrameLayout.LayoutParams.MATCH_PARENT
            )
        )
        cameraFrame.addView(
            cameraPlaceholder,
            FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                FrameLayout.LayoutParams.MATCH_PARENT
            )
        )
        outer.addView(
            cameraFrame,
            LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                0,
                1.25f
            ).apply {
                topMargin = 18
                bottomMargin = 18
            }
        )

        val controls = LinearLayout(context).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER
        }

        pushToTalkButton = Button(context).apply {
            text = "Hold Mic"
            setOnTouchListener { view, event ->
                when (event.actionMasked) {
                    MotionEvent.ACTION_DOWN -> {
                        onPushToTalkStart()
                        true
                    }

                    MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                        onPushToTalkStop()
                        view.performClick()
                        true
                    }

                    else -> false
                }
            }
        }
        memoryButton = Button(context).apply {
            text = "Memories (0)"
            setOnClickListener { onMemories() }
        }
        controls.addView(
            pushToTalkButton,
            LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f).apply {
                marginEnd = 12
            }
        )
        controls.addView(
            memoryButton,
            LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
        )
        outer.addView(controls)

        questionInput = EditText(context).apply {
            hint = "Ask what the camera sees..."
            minLines = 1
            maxLines = 3
            setSingleLine(false)
        }
        outer.addView(
            questionInput,
            LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply {
                topMargin = 14
            }
        )

        val askButton = Button(context).apply {
            text = "Ask Offline"
            setOnClickListener { onAsk(questionInput.text.toString()) }
        }
        outer.addView(askButton)

        val answerScroll = ScrollView(context)
        answerText = TextView(context).apply {
            textSize = 16f
            setTextColor(context.color(R.color.snapon_ink))
            setPadding(0, 16, 0, 16)
        }
        answerScroll.addView(answerText)
        outer.addView(
            answerScroll,
            LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                0,
                0.75f
            )
        )

        root = outer
    }

    fun showStatus(message: String) {
        statusText.text = message
    }

    fun showAnswer(message: String) {
        answerText.text = message
    }

    fun setQuestion(text: String) {
        questionInput.setText(text)
        questionInput.setSelection(questionInput.text.length)
    }

    fun showCameraPlaceholder(message: String) {
        cameraPlaceholder.text = message
        cameraPlaceholder.visibility = View.VISIBLE
    }

    fun hideCameraPlaceholder() {
        cameraPlaceholder.visibility = View.GONE
    }

    fun setListening(isListening: Boolean) {
        pushToTalkButton.text = if (isListening) "Listening" else "Hold Mic"
    }

    fun setMemoryCount(count: Int) {
        memoryButton.text = "Memories ($count)"
    }

    private fun Context.color(id: Int): Int {
        return getColor(id)
    }
}
