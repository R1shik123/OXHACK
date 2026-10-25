package app.sightline

import android.content.Context
import android.content.pm.ApplicationInfo
import android.content.pm.PackageManager
import android.os.Build
import java.io.File

/**
 * Refuse to run in environments where our guarantees don't hold: a re-signed (tampered) APK,
 * a debuggable build, or a rooted device (where other apps could read our memory and camera).
 */
object Integrity {
    fun problem(ctx: Context): String? {
        if (ctx.applicationInfo.flags and ApplicationInfo.FLAG_DEBUGGABLE != 0) return "Debug builds are not allowed."
        val expected = BuildConfig.SIGNER_SHA256.lowercase()
        if (expected.isNotEmpty()) {
            val info = ctx.packageManager.getPackageInfo(ctx.packageName, PackageManager.GET_SIGNING_CERTIFICATES)
            val signers = info.signingInfo?.apkContentsSigners ?: emptyArray()
            if (signers.none { Pinning.sha256Hex(it.toByteArray()) == expected }) return "This copy of the app was modified (signature mismatch)."
        }
        if (rooted()) return "This device appears to be rooted. FindMyTeam won't run on rooted devices."
        return null
    }

    private fun rooted(): Boolean {
        if (Build.TAGS?.contains("test-keys") == true) return true
        val paths = listOf("/system/bin/su", "/system/xbin/su", "/sbin/su", "/su/bin/su", "/data/local/xbin/su", "/data/local/bin/su",
            "/system/sd/xbin/su", "/system/app/Superuser.apk", "/data/adb/magisk")
        return paths.any { File(it).exists() }
    }
}
