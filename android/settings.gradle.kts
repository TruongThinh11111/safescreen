pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}
dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
        // Tesseract4Android chỉ phát hành trên JitPack
        maven("https://jitpack.io")
    }
}

rootProject.name = "SafeScreen"
include(":app")
