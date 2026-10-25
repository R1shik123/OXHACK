package app.sightline

import android.opengl.GLES11Ext
import android.opengl.GLES20
import com.google.ar.core.Coordinates2d
import com.google.ar.core.Frame
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

/** Draws the ARCore camera image as a full-screen quad (external OES texture). */
class BackgroundRenderer {
    var textureId = -1
        private set
    private var program = 0
    private val quad: FloatBuffer = floatBuffer(floatArrayOf(-1f, -1f, 1f, -1f, -1f, 1f, 1f, 1f))
    private val uvs: FloatBuffer = floatBuffer(FloatArray(8))

    fun create() {
        val t = IntArray(1)
        GLES20.glGenTextures(1, t, 0)
        textureId = t[0]
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, textureId)
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE)
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE)
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR)
        program = link(
            """
            attribute vec4 a_Position; attribute vec2 a_TexCoord; varying vec2 v_TexCoord;
            void main() { gl_Position = a_Position; v_TexCoord = a_TexCoord; }
            """,
            """
            #extension GL_OES_EGL_image_external : require
            precision mediump float; varying vec2 v_TexCoord; uniform samplerExternalOES u_Texture;
            void main() { gl_FragColor = texture2D(u_Texture, v_TexCoord); }
            """,
        )
    }

    fun draw(frame: Frame) {
        if (frame.hasDisplayGeometryChanged()) {
            quad.position(0)
            uvs.position(0)
            frame.transformCoordinates2d(Coordinates2d.OPENGL_NORMALIZED_DEVICE_COORDINATES, quad, Coordinates2d.TEXTURE_NORMALIZED, uvs)
        }
        if (frame.timestamp == 0L) return
        GLES20.glDisable(GLES20.GL_DEPTH_TEST)
        GLES20.glDepthMask(false)
        GLES20.glUseProgram(program)
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, textureId)
        val pos = GLES20.glGetAttribLocation(program, "a_Position")
        val uv = GLES20.glGetAttribLocation(program, "a_TexCoord")
        quad.position(0)
        uvs.position(0)
        GLES20.glVertexAttribPointer(pos, 2, GLES20.GL_FLOAT, false, 0, quad)
        GLES20.glVertexAttribPointer(uv, 2, GLES20.GL_FLOAT, false, 0, uvs)
        GLES20.glEnableVertexAttribArray(pos)
        GLES20.glEnableVertexAttribArray(uv)
        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4)
        GLES20.glDisableVertexAttribArray(pos)
        GLES20.glDisableVertexAttribArray(uv)
        GLES20.glDepthMask(true)
        GLES20.glEnable(GLES20.GL_DEPTH_TEST)
    }

    private fun floatBuffer(a: FloatArray): FloatBuffer =
        ByteBuffer.allocateDirect(a.size * 4).order(ByteOrder.nativeOrder()).asFloatBuffer().apply { put(a); position(0) }

    private fun link(vs: String, fs: String): Int {
        fun shader(type: Int, src: String) = GLES20.glCreateShader(type).also {
            GLES20.glShaderSource(it, src)
            GLES20.glCompileShader(it)
        }
        return GLES20.glCreateProgram().also {
            GLES20.glAttachShader(it, shader(GLES20.GL_VERTEX_SHADER, vs))
            GLES20.glAttachShader(it, shader(GLES20.GL_FRAGMENT_SHADER, fs))
            GLES20.glLinkProgram(it)
        }
    }
}
