import java.util.Properties

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

// Release signing material lives outside version control (android/keystore.properties, mode 0600).
val ks = Properties().apply {
    val f = rootProject.file("keystore.properties")
    if (f.exists()) f.inputStream().use { load(it) }
}

android {
    namespace = "app.sightline"
    compileSdk = 35

    defaultConfig {
        applicationId = "app.sightline"
        minSdk = 29 // TLS 1.3 everywhere, modern scoped storage, no legacy APIs
        targetSdk = 35
        versionCode = 2
        versionName = "2.0"
        // SHA-256 of our release signing certificate: the app refuses to run if re-signed by someone else.
        buildConfigField("String", "SIGNER_SHA256", "\"${ks.getProperty("signerSha256", "")}\"")
    }

    signingConfigs {
        create("release") {
            if (ks.isNotEmpty()) {
                storeFile = rootProject.file(ks.getProperty("storeFile"))
                storePassword = ks.getProperty("storePassword")
                keyAlias = ks.getProperty("keyAlias")
                keyPassword = ks.getProperty("keyPassword")
                enableV1Signing = false
                enableV2Signing = true
                enableV3Signing = true
            }
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = true
            isShrinkResources = true
            isDebuggable = false
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
            signingConfig = signingConfigs.getByName("release")
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
    buildFeatures { buildConfig = true }
    packaging { resources { excludes += setOf("META-INF/*.version", "META-INF/*.kotlin_module", "kotlin/**", "DebugProbesKt.bin") } }
    dependenciesInfo { includeInApk = false; includeInBundle = false }
}

dependencies {
    implementation("com.google.ar:core:1.44.0")
    implementation("com.squareup.okhttp3:okhttp:4.12.0")
    implementation("com.google.zxing:core:3.5.3") // offline QR decoding, no Google services involved
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("androidx.core:core-ktx:1.13.1")
}
