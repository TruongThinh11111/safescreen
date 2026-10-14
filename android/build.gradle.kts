plugins {
    id("com.android.application") version "9.4.1" apply false
    // Plugin biên dịch Compose: phiên bản PHẢI trùng phiên bản Kotlin mà AGP 9.4.1 dùng (2.2.10)
    id("org.jetbrains.kotlin.plugin.compose") version "2.2.10" apply false
}
