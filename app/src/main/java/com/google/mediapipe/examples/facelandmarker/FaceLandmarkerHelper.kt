/*
 * Copyright 2023 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *             http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.google.mediapipe.examples.facelandmarker

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.os.SystemClock
import android.util.Log
import androidx.annotation.VisibleForTesting
import androidx.camera.core.ImageProxy
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.framework.image.MPImage
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarkerResult
import kotlin.math.pow
import kotlin.math.sqrt


class FaceLandmarkerHelper(
    var minFaceDetectionConfidence: Float = DEFAULT_FACE_DETECTION_CONFIDENCE,
    var minFaceTrackingConfidence: Float = DEFAULT_FACE_TRACKING_CONFIDENCE,
    var minFacePresenceConfidence: Float = DEFAULT_FACE_PRESENCE_CONFIDENCE,
    var maxNumFaces: Int = DEFAULT_NUM_FACES,
    var currentDelegate: Int = DELEGATE_CPU,
    var runningMode: RunningMode = RunningMode.IMAGE,
    val context: Context,
    // this listener is only used when running in RunningMode.LIVE_STREAM
    val faceLandmarkerHelperListener: LandmarkerListener? = null
) {

    // For this example this needs to be a var so it can be reset on changes.
    // If the Face Landmarker will not change, a lazy val would be preferable.
    private var faceLandmarker: FaceLandmarker? = null

    init {
        setupFaceLandmarker()
    }

    fun clearFaceLandmarker() {
        faceLandmarker?.close()
        faceLandmarker = null
    }

    // Return running status of FaceLandmarkerHelper
    fun isClose(): Boolean {
        return faceLandmarker == null
    }

    // Initialize the Face landmarker using current settings on the
    // thread that is using it. CPU can be used with Landmarker
    // that are created on the main thread and used on a background thread, but
    // the GPU delegate needs to be used on the thread that initialized the
    // Landmarker
    fun setupFaceLandmarker() {
        // Set general face landmarker options
        val baseOptionBuilder = BaseOptions.builder()

        // Use the specified hardware for running the model. Default to CPU
        when (currentDelegate) {
            DELEGATE_CPU -> {
                baseOptionBuilder.setDelegate(Delegate.CPU)
            }
            DELEGATE_GPU -> {
                baseOptionBuilder.setDelegate(Delegate.GPU)
            }
        }

        baseOptionBuilder.setModelAssetPath(MP_FACE_LANDMARKER_TASK)

        // Check if runningMode is consistent with faceLandmarkerHelperListener
        when (runningMode) {
            RunningMode.LIVE_STREAM -> {
                if (faceLandmarkerHelperListener == null) {
                    throw IllegalStateException(
                        "faceLandmarkerHelperListener must be set when runningMode is LIVE_STREAM."
                    )
                }
            }
            else -> {
                // no-op
            }
        }

        try {
            val baseOptions = baseOptionBuilder.build()
            // Create an option builder with base options and specific
            // options only use for Face Landmarker.
            val optionsBuilder =
                FaceLandmarker.FaceLandmarkerOptions.builder()
                    .setBaseOptions(baseOptions)
                    .setMinFaceDetectionConfidence(minFaceDetectionConfidence)
                    .setMinTrackingConfidence(minFaceTrackingConfidence)
                    .setMinFacePresenceConfidence(minFacePresenceConfidence)
                    .setNumFaces(maxNumFaces)
                    .setRunningMode(runningMode)

            // The ResultListener and ErrorListener only use for LIVE_STREAM mode.
            if (runningMode == RunningMode.LIVE_STREAM) {
                optionsBuilder
                    .setResultListener(this::returnLivestreamResult)
                    .setErrorListener(this::returnLivestreamError)
            }

            val options = optionsBuilder.build()
            faceLandmarker =
                FaceLandmarker.createFromOptions(context, options)
        } catch (e: IllegalStateException) {
            faceLandmarkerHelperListener?.onError(
                "Face Landmarker failed to initialize. See error logs for " +
                        "details"
            )
            Log.e(
                TAG, "MediaPipe failed to load the task with error: " + e
                    .message
            )
        } catch (e: RuntimeException) {
            // This occurs if the model being used does not support GPU
            faceLandmarkerHelperListener?.onError(
                "Face Landmarker failed to initialize. See error logs for " +
                        "details", GPU_ERROR
            )
            Log.e(
                TAG,
                "Face Landmarker failed to load model with error: " + e.message
            )
        }
    }

    // Convert the ImageProxy to MP Image and feed it to FacelandmakerHelper.
    fun detectLiveStream(
        imageProxy: ImageProxy,
        isFrontCamera: Boolean
    ) {
        if (runningMode != RunningMode.LIVE_STREAM) {
            throw IllegalArgumentException(
                "Attempting to call detectLiveStream" +
                        " while not using RunningMode.LIVE_STREAM"
            )
        }
        val frameTime = SystemClock.uptimeMillis()

        // Copy out RGB bits from the frame to a bitmap buffer
        val bitmapBuffer =
            Bitmap.createBitmap(
                imageProxy.width,
                imageProxy.height,
                Bitmap.Config.ARGB_8888
            )
        imageProxy.use { bitmapBuffer.copyPixelsFromBuffer(imageProxy.planes[0].buffer) }
        imageProxy.close()

        val matrix = Matrix().apply {
            // Rotate the frame received from the camera to be in the same direction as it'll be shown
            postRotate(imageProxy.imageInfo.rotationDegrees.toFloat())

            // flip image if user use front camera
            if (isFrontCamera) {
                postScale(
                    -1f,
                    1f,
                    imageProxy.width.toFloat(),
                    imageProxy.height.toFloat()
                )
            }
        }
        val rotatedBitmap = Bitmap.createBitmap(
            bitmapBuffer, 0, 0, bitmapBuffer.width, bitmapBuffer.height,
            matrix, true
        )

        // Convert the input Bitmap object to an MPImage object to run inference
        val mpImage = BitmapImageBuilder(rotatedBitmap).build()

        detectAsync(mpImage, frameTime)
    }

    // Run face face landmark using MediaPipe Face Landmarker API
    @VisibleForTesting
    fun detectAsync(mpImage: MPImage, frameTime: Long) {
        faceLandmarker?.detectAsync(mpImage, frameTime)
        // As we're using running mode LIVE_STREAM, the landmark result will
        // be returned in returnLivestreamResult function
    }


    private fun returnLivestreamResult(
        result: FaceLandmarkerResult,
        input: MPImage
    ) {

        var avgEAR = 0.1
        if (result.faceLandmarks().size > 0) {

            val finishTimeMs = SystemClock.uptimeMillis()
            val inferenceTime = finishTimeMs - result.timestampMs()

            val detectedFaces = result.faceLandmarks()
            for (face in detectedFaces) {
                // Get the right eye landmarks
                val rightEyeLandmark33 = face[33].x() to face[33].y()  // Landmark 33
                val rightEyeLandmark133 = face[133].x() to face[133].y() // Landmark 133

                // Calculate the horizontal distance between landmarks 33 and 133 for the right eye
                val rightHorizontalDist = euclideanDistance(rightEyeLandmark33.first, rightEyeLandmark33.second, rightEyeLandmark133.first, rightEyeLandmark133.second)

                // Calculate the vertical distance between landmarks 249 and 146 for the right eye
                val rightEyeLandmark159 = face[159].x() to face[159].y()  // Landmark 249
                val rightEyeLandmark145 = face[145].x() to face[145].y() // Landmark 146
                val rightVerticalDist = euclideanDistance(rightEyeLandmark159.first, rightEyeLandmark159.second, rightEyeLandmark145.first, rightEyeLandmark145.second)

                // Calculate the EAR for the right eye
                val rightEAR = rightVerticalDist / rightHorizontalDist


                // Get the left eye landmarks
                val leftEyeLandmark263 = face[263].x() to face[263].y()  // Landmark 263
                val leftEyeLandmark362 = face[362].x() to face[362].y() // Landmark 362

                // Calculate the horizontal distance between landmarks 263 and 362 for the left eye
                val leftHorizontalDist = euclideanDistance(leftEyeLandmark263.first, leftEyeLandmark263.second, leftEyeLandmark362.first, leftEyeLandmark362.second)

                // Calculate the vertical distance between landmarks 386 and 374 for the left eye
                val leftEyeLandmark386 = face[386].x() to face[386].y()  // Landmark 386
                val leftEyeLandmark374 = face[373].x() to face[373].y() // Landmark 374
                val leftVerticalDist = euclideanDistance(leftEyeLandmark386.first, leftEyeLandmark386.second, leftEyeLandmark374.first, leftEyeLandmark374.second)

                // Calculate the EAR for the left eye
                val leftEAR = leftVerticalDist / leftHorizontalDist



                // Calculate the average EAR for both eyes
                avgEAR = (rightEAR + leftEAR) / 2.0

                // Log the average EAR
                Log.d(TAG, "Average EAR: $avgEAR")

            }

            faceLandmarkerHelperListener?.onResults(
                ResultBundle(
                    result,
                    inferenceTime,
                    input.height,
                    input.width,
                    avgEAR
                )
            )
        } else {
            faceLandmarkerHelperListener?.onEmpty()
        }
    }


    // Function to calculate the Euclidean distance between two points (landmarks)
    private fun euclideanDistance(x1: Float, y1: Float, x2: Float, y2: Float): Double {
        return sqrt((x2 - x1).toDouble().pow(2.0) + (y2 - y1).toDouble().pow(2.0))
    }








    // Return errors thrown during detection to this FaceLandmarkerHelper's
    // caller
    private fun returnLivestreamError(error: RuntimeException) {
        faceLandmarkerHelperListener?.onError(
            error.message ?: "An unknown error has occurred"
        )
    }

    companion object {
        const val TAG = "FaceLandmarkerHelper"
        private const val MP_FACE_LANDMARKER_TASK = "face_landmarker.task"

        const val DELEGATE_CPU = 0
        const val DELEGATE_GPU = 1
        const val DEFAULT_FACE_DETECTION_CONFIDENCE = 0.5F
        const val DEFAULT_FACE_TRACKING_CONFIDENCE = 0.5F
        const val DEFAULT_FACE_PRESENCE_CONFIDENCE = 0.5F
        const val DEFAULT_NUM_FACES = 1
        const val OTHER_ERROR = 0
        const val GPU_ERROR = 1
    }

    data class ResultBundle(
        val result: FaceLandmarkerResult,
        val inferenceTime: Long,
        val inputImageHeight: Int,
        val inputImageWidth: Int,
        val avgEAR : Double
    )

    data class VideoResultBundle(
        val results: List<FaceLandmarkerResult>,
        val inferenceTime: Long,
        val inputImageHeight: Int,
        val inputImageWidth: Int,
    )

    interface LandmarkerListener {
        fun onError(error: String, errorCode: Int = OTHER_ERROR)
        fun onResults(resultBundle: ResultBundle)

        fun onEmpty() {}
    }
}
