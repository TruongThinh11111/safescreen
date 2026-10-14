plugins {
    id("com.android.application")
    // AGP 9 đã tích hợp sẵn Kotlin → không cần plugin org.jetbrains.kotlin.android
    id("org.jetbrains.kotlin.plugin.compose")
}

android {
    namespace = "com.safescreen"
    compileSdk = 37

    defaultConfig {
        applicationId = "com.safescreen"
        minSdk = 26          // java.time + named group trong java.util.regex cần API 26
        targetSdk = 37
        versionCode = 1
        versionName = "0.1"
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    buildTypes {
        release {
            // Chỉ đóng gói thư viện C++ (Tesseract) cho CPU arm64 — gần như mọi điện thoại Android hiện nay.
            // Bản debug vẫn giữ đủ 4 loại CPU để chạy được trên máy ảo x86_64.
            ndk {
                abiFilters += "arm64-v8a"
            }
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    buildFeatures {
        compose = true   // bật Jetpack Compose cho module app
    }
}

dependencies {
    // Chỉ dùng API JsonElement (không cần compiler plugin): giữ thứ tự khóa, chạy giống nhau trên JVM và Android
    implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.9.0")

    // Compose: BOM quy định phiên bản chung cho mọi thư viện androidx.compose.* bên dưới
    implementation(platform("androidx.compose:compose-bom:2026.02.01"))
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.material3:material3")
    debugImplementation("androidx.compose.ui:ui-tooling")   // cho @Preview trong Android Studio
    implementation("androidx.compose.ui:ui-tooling-preview")
    // setContent {} + rememberLauncherForActivityResult (Photo Picker)
    implementation("androidx.activity:activity-compose:1.13.0")
    // viewModel(): giữ trạng thái + bộ máy Tesseract khi xoay màn hình
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.11.0")
    // LifecycleResumeEffect: đọc lại trạng thái quyền mỗi khi người dùng quay lại app từ Cài đặt
    implementation("androidx.lifecycle:lifecycle-runtime-compose:2.11.0")

    // OCR: Tesseract (C++) đóng gói cho Android. Bản "standard" dùng 1 luồng, giống OMP_THREAD_LIMIT=1 bên Python
    implementation("cz.adaptech.tesseract4android:tesseract4android:4.9.0")

    testImplementation("junit:junit:4.13.2")
    androidTestImplementation("androidx.test.ext:junit:1.1.5")
    androidTestImplementation("androidx.test:runner:1.5.2")
}

// Chép mô hình từ ml/models (do safescreen/export_android.py tạo ra). Chạy lại mỗi khi train/export lại:
//   gradlew :app:syncModels
val mlModels = rootProject.file("../ml/models")
tasks.register<Copy>("syncModelAssets") {
    from(mlModels) { include("crf_model.json", "config.json") }
    into(layout.projectDirectory.dir("src/main/assets"))
}
tasks.register<Copy>("syncGoldenTest") {
    from(mlModels) { include("golden_test.json") }
    into(layout.projectDirectory.dir("src/androidTest/assets"))
}
tasks.register("syncModels") {
    group = "safescreen"
    description = "Chép crf_model.json, config.json, golden_test.json từ ml/models"
    dependsOn("syncModelAssets", "syncGoldenTest")
}
