plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.snapon.mobile"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.snapon.mobile"
        minSdk = 26
        targetSdk = 34
        versionCode = 1
        versionName = "0.1.0"
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_1_8
        targetCompatibility = JavaVersion.VERSION_1_8
    }

    kotlinOptions {
        jvmTarget = "1.8"
    }

    sourceSets {
        getByName("main") {
            assets.srcDirs("../config", "../models")
            java.srcDir("../android/runtime")
        }
    }
}

dependencies {
    val cameraXVersion = "1.3.1"

    implementation("androidx.activity:activity-ktx:1.8.2")
    implementation("androidx.appcompat:appcompat:1.6.1")
    implementation("androidx.arch.core:core-runtime:2.2.0")
    implementation("androidx.core:core-ktx:1.12.0")
    implementation("androidx.customview:customview:1.1.0")
    implementation("androidx.drawerlayout:drawerlayout:1.1.1")
    implementation("androidx.lifecycle:lifecycle-common:2.7.0")
    implementation("androidx.lifecycle:lifecycle-livedata:2.7.0")
    implementation("androidx.lifecycle:lifecycle-livedata-core:2.7.0")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.7.0")
    implementation("androidx.lifecycle:lifecycle-viewmodel-ktx:2.7.0")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.7.3")
    implementation("androidx.camera:camera-camera2:$cameraXVersion")
    implementation("androidx.camera:camera-lifecycle:$cameraXVersion")
    implementation("androidx.camera:camera-view:$cameraXVersion")
    implementation(project(":android-foundation"))

    // ExecuTorch on-device runtime (XNNPACK CPU backend). Version must match
    // the executorch pip package used to produce models/xnnpack/*.pte
    // (currently 1.3.1) — mismatched runtime/artifact versions can fail to
    // load or silently misbehave. Verify this resolves against Maven Central
    // when building; if 1.3.1 isn't published, use the closest available
    // release and re-check compatibility.
    implementation("org.pytorch:executorch-android:1.3.1")
}
