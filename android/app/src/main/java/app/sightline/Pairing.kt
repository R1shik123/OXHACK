package app.sightline

import android.net.Uri
import okhttp3.ConnectionSpec
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.TlsVersion
import org.json.JSONObject
import java.security.MessageDigest
import java.security.SecureRandom
import java.security.cert.CertificateException
import java.security.cert.X509Certificate
import java.util.concurrent.TimeUnit
import javax.net.ssl.SSLContext
import javax.net.ssl.X509TrustManager

/** Server + device token. Lives in memory for one app session only: never written anywhere. */
data class Pairing(val host: String, val port: Int, val fingerprint: String, val deviceId: String, val token: String, val name: String) {
    val httpBase get() = "https://$host:$port"
    val wsUrl get() = "wss://$host:$port/ws/phone"
}

/** sightline://pair?h=..&p=..&c=..&f=.. from the commander's QR code. */
data class PairLink(val host: String, val port: Int, val code: String, val fingerprint: String) {
    companion object {
        private val HOST = Regex("^[A-Za-z0-9.:-]{1,253}$")
        private val HEX64 = Regex("^[0-9a-f]{64}$")

        /** Parses the text of a scanned QR code. */
        fun parse(text: String?): PairLink? {
            if (text == null || text.length > 512) return null
            val uri = runCatching { Uri.parse(text) }.getOrNull() ?: return null
            if (uri.scheme != "sightline" || uri.host != "pair") return null
            val h = uri.getQueryParameter("h") ?: return null
            val p = uri.getQueryParameter("p")?.toIntOrNull() ?: return null
            val c = uri.getQueryParameter("c") ?: return null
            val f = uri.getQueryParameter("f")?.lowercase() ?: return null
            if (!HOST.matches(h) || p !in 1..65535 || c.length !in 8..64 || !HEX64.matches(f)) return null
            return PairLink(h, p, c, f)
        }
    }
}

/**
 * OkHttp client that trusts exactly one certificate: the one whose SHA-256 came in the QR code.
 * Hostname checks are skipped on purpose: the server is reached by LAN IP and identity comes
 * from the pinned certificate itself, which is a stronger check than any CA chain here.
 */
object Pinning {
    fun sha256Hex(bytes: ByteArray): String = MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }

    fun client(fingerprint: String): OkHttpClient {
        val tm = object : X509TrustManager {
            override fun checkClientTrusted(chain: Array<X509Certificate>, authType: String) = throw CertificateException("no client auth")
            override fun checkServerTrusted(chain: Array<X509Certificate>, authType: String) {
                val leaf = chain.firstOrNull() ?: throw CertificateException("empty chain")
                if (sha256Hex(leaf.encoded) != fingerprint) throw CertificateException("certificate does not match pairing")
            }
            override fun getAcceptedIssuers(): Array<X509Certificate> = emptyArray()
        }
        val ssl = SSLContext.getInstance("TLSv1.3").apply { init(null, arrayOf(tm), SecureRandom()) }
        val tls13 = ConnectionSpec.Builder(ConnectionSpec.RESTRICTED_TLS).tlsVersions(TlsVersion.TLS_1_3).build()
        return OkHttpClient.Builder()
            .connectionSpecs(listOf(tls13)) // TLS 1.3 only, no fallback
            .sslSocketFactory(ssl.socketFactory, tm)
            .hostnameVerifier { _, _ -> true }
            .connectTimeout(8, TimeUnit.SECONDS)
            .pingInterval(5, TimeUnit.SECONDS)
            .build()
    }

    /** Redeem a one-time code for a device token (blocking: call off the main thread). */
    fun redeem(link: PairLink, name: String): Result<Pairing> = runCatching {
        val body = JSONObject().put("code", link.code).put("name", name).put("platform", "android").toString()
        val req = Request.Builder().url("https://${link.host}:${link.port}/api/pair")
            .post(body.toRequestBody("application/json".toMediaType())).build()
        client(link.fingerprint).newCall(req).execute().use { r ->
            when (r.code) {
                200 -> {
                    val j = JSONObject(r.body!!.string())
                    Pairing(link.host, link.port, link.fingerprint, j.getString("device_id"), j.getString("token"), j.optString("name", name))
                }
                403 -> error("That QR code was already used or has expired. Make a new one on the laptop.")
                429 -> error("Too many attempts. Wait a minute.")
                else -> error("Server said ${r.code}")
            }
        }
    }
}
