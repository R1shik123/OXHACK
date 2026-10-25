-keep class com.google.ar.** { *; }
-dontwarn okhttp3.internal.platform.**
-dontwarn org.conscrypt.**
-dontwarn org.bouncycastle.**
-dontwarn org.openjsse.**
# Strip every log call from release builds: nothing (tokens, addresses) can leak into logcat.
-assumenosideeffects class android.util.Log { *; }
-repackageclasses ''
-allowaccessmodification
