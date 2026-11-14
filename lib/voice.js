/**
 * QQ 语音条本地转写（零 token、零联网）。
 *
 * 链路：
 *   QQ 语音（SILK v3 / amr / wav …）
 *     → SILK 用 silk_v3_decoder 解成 PCM（非 SILK 跳过）
 *     → ffmpeg 转 16k 单声道 wav
 *     → SenseVoiceSmall（sherpa-onnx，纯 CPU）识别
 *
 * 之前用 whisper.cpp：需要 whisper-cli.exe + ggml-*.bin，本机根本没有，
 * 每次语音都报 missing-tools。现在改用 DSH 自带的 sherpa-onnx 原生模块
 * 与 speech-to-text 插件已经下载好的 SenseVoice 模型（模型文件就在
 * ~/.dsh/speech-to-text/sensevoice/models 下），不再需要额外安装。
 *
 * 全部子进程隐藏窗口运行；可执行文件/模型路径可在构造时覆盖。
 */

import { spawn } from 'node:child_process'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { createRequire } from 'node:module'

export const SILK_MAGIC = Buffer.from('#!SILK_V3', 'latin1')

export const VOICE_EXTS = new Set([
  '.amr', '.silk', '.slk', '.wav', '.mp3', '.m4a', '.aac',
  '.ogg', '.opus', '.flac', '.wma', '.pcm', '.spx',
])

const DSH_HOME = () => {
  const raw = process.env.DSH_HOME
  if (raw && raw.trim()) return raw.trim()
  return path.join(process.env.USERPROFILE ?? os.homedir(), '.dsh')
}

/**
 * SenseVoice + Silero VAD 的模型文件（由 dsh-experimental-speech-to-text-sensevoice
 * 插件下载并校验过，这里直接复用，不再重复下载 240MB 模型）。
 */
export const SENSEVOICE_MODEL_DIRS = [
  path.join(DSH_HOME(), 'speech-to-text', 'sensevoice', 'models', 'sensevoice-onnx'),
  path.join(DSH_HOME(), 'speech-to-text', 'sensevoice', 'models'),
]

export const SILERO_MODEL_CANDIDATES = [
  path.join(DSH_HOME(), 'speech-to-text', 'sensevoice', 'models', 'silero', 'silero_vad.onnx'),
  path.join(DSH_HOME(), 'speech-to-text', 'silero', 'silero_vad.onnx'),
]

/**
 * sherpa-onnx 原生模块的候选位置。DSH 把原生扩展放在 app.asar.unpacked 下，
 * 从插件目录 require('sherpa-onnx-node') 是解析不到的，只能直连 .node。
 */
export const SHERPA_NATIVE_CANDIDATES = [
  // DSH 安装目录（asar.unpacked）
  ...(process.resourcesPath
    ? [path.join(process.resourcesPath, 'app.asar.unpacked', 'dsh', 'node_modules', 'sherpa-onnx-win-x64', 'sherpa-onnx.node')]
    : []),
  path.join(process.env.ProgramFiles ?? 'C:\\Program Files', '(x86)', 'deepseekHraness', 'resources',
    'app.asar.unpacked', 'dsh', 'node_modules', 'sherpa-onnx-win-x64', 'sherpa-onnx.node'),
  'D:\\Program Files (x86)\\deepseekHraness\\resources\\app.asar.unpacked\\dsh\\node_modules\\sherpa-onnx-win-x64\\sherpa-onnx.node',
  path.join(DSH_HOME(), 'qq-console-v2', 'bin', 'sherpa-onnx.node'),
]

export const DEFAULT_VOICE_TOOLS = {
  silkDecoder: path.join(DSH_HOME(), 'qq-console-v2', 'bin', 'silk_v3_decoder.exe'),
  ffmpeg: path.join(DSH_HOME(), 'qq-console-v2', 'bin', 'ffmpeg.exe'),
  senseVoiceModel: path.join(SENSEVOICE_MODEL_DIRS[0], 'model.int8.onnx'),
  senseVoiceTokens: path.join(SENSEVOICE_MODEL_DIRS[0], 'tokens.txt'),
  sileroVad: SILERO_MODEL_CANDIDATES[0],
  sherpaNative: '',
}

const HALLUCINATION_RE = new RegExp(
  '^(请不吝点赞|点赞|订阅|转发|打赏|明镜与点点|字幕由|字幕志愿者|' +
  '感谢观看|谢谢观看|谢谢大家|下期再见|再见|bye|thank you|thanks|' +
  'please subscribe|subscribe)[。.!！?？\\s]*$',
  'i',
)

/** 按 MIME、扩展名或文件头判断是不是语音附件。 */
export function isVoiceAttachment({ name = '', contentType = '', buf = null } = {}) {
  if (/^audio\//i.test(String(contentType))) return true
  if (VOICE_EXTS.has(path.extname(String(name)).toLowerCase())) return true
  if (Buffer.isBuffer(buf) && buf.subarray(0, 128).includes(SILK_MAGIC)) return true
  return false
}

function firstExisting(candidates, suffixHint = '') {
  for (const candidate of candidates) {
    try {
      if (!candidate) continue
      if (suffixHint && !candidate.endsWith(suffixHint)) continue
      if (fs.statSync(candidate).isFile()) return candidate
    } catch { /* 下一个 */ }
  }
  return ''
}

function run(file, args, { timeoutMs = 180_000 } = {}) {
  return new Promise((resolve) => {
    let settled = false
    const child = spawn(file, args, {
      windowsHide: true,
      stdio: ['ignore', 'pipe', 'pipe'],
    })
    let stdout = ''
    let stderr = ''
    child.stdout?.setEncoding('utf8')
    child.stderr?.setEncoding('utf8')
    child.stdout?.on('data', (chunk) => { stdout += chunk })
    child.stderr?.on('data', (chunk) => { stderr += chunk })
    const timer = setTimeout(() => {
      if (settled) return
      settled = true
      try { child.kill('SIGKILL') } catch { /* ignore */ }
      resolve({ code: -1, stdout, stderr: `${stderr}\n超时（${timeoutMs}ms）` })
    }, timeoutMs)
    timer.unref?.()
    child.on('error', (err) => {
      if (settled) return
      settled = true
      clearTimeout(timer)
      resolve({ code: -1, stdout, stderr: err?.message ?? String(err) })
    })
    child.on('close', (code) => {
      if (settled) return
      settled = true
      clearTimeout(timer)
      resolve({ code: Number(code ?? -1), stdout, stderr })
    })
  })
}

function readHead(file, length = 128) {
  let fd
  try {
    fd = fs.openSync(file, 'r')
    const buf = Buffer.alloc(length)
    const n = fs.readSync(fd, buf, 0, length, 0)
    return buf.subarray(0, n)
  } catch {
    return Buffer.alloc(0)
  } finally {
    if (fd !== undefined) {
      try { fs.closeSync(fd) } catch { /* ignore */ }
    }
  }
}

function validText(text) {
  const value = String(text ?? '').trim()
  // SenseVoice 有时会把整条静音/噪音标成一串标点，这里只要"有字"的。
  if (!/[\p{L}\p{N}]/u.test(value)) return false
  if (value.replace(/[\p{P}\p{S}\s]/gu, '').length < 2) return false
  if (HALLUCINATION_RE.test(value)) return false
  return true
}

/** 只留文字部分：去掉 SenseVoice 的 <|zh|> / <|NEUTRAL|> 这类标签。 */
function cleanSenseVoiceText(text) {
  return String(text ?? '')
    .replace(/<\|[^|]*\|>/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
}

let sherpaCache = { file: '', mod: null }

// SenseVoice 单次稳定吃下约 30 秒；取 28 秒做整段识别的上限，超过就定长窗口切。
const MAX_WHOLE_SAMPLES = 28 * 16000

/**
 * 自己解析 16k 单声道 wav，而不是用 sherpa.readWave()。
 *
 * sherpa 的 readWave 返回的 Float32Array 建在**原生层分配的外部 ArrayBuffer**
 * 上；把它再喂回原生层时，Electron 宿主会直接抛
 * `External buffers are not allowed`（普通 node 里不报，所以本地脚本测不出来）。
 * 这里用 fs 读文件、自己拆 RIFF，得到的 Float32Array 自带普通内存，绝对安全。
 *
 * ffmpeg 那边固定输出 pcm_s16le / 16k / 单声道，所以主路径是 16 位整数 PCM；
 * 顺带兼容 32 位浮点与 8/24/32 位整数，避免以后换参数就崩。
 */
export function readWavSamples(file) {
  const buf = fs.readFileSync(file)
  if (buf.length < 44) throw new Error('wav 太小')
  if (buf.toString('ascii', 0, 4) !== 'RIFF' || buf.toString('ascii', 8, 12) !== 'WAVE') {
    throw new Error('不是 wav 文件')
  }
  let offset = 12
  let format = 1
  let channels = 1
  let sampleRate = 16000
  let bits = 16
  let dataStart = -1
  let dataSize = 0
  while (offset + 8 <= buf.length) {
    const id = buf.toString('ascii', offset, offset + 4)
    const size = buf.readUInt32LE(offset + 4)
    const body = offset + 8
    if (id === 'fmt ') {
      format = buf.readUInt16LE(body)
      channels = buf.readUInt16LE(body + 2) || 1
      sampleRate = buf.readUInt32LE(body + 4) || 16000
      bits = buf.readUInt16LE(body + 14) || 16
    } else if (id === 'data') {
      dataStart = body
      dataSize = Math.min(size, buf.length - body)
      break
    }
    offset = body + size + (size % 2)
  }
  if (dataStart < 0) throw new Error('wav 里没有 data 块')
  const bytesPerSample = Math.max(1, Math.floor(bits / 8))
  const frames = Math.floor(dataSize / (bytesPerSample * channels))
  const out = new Float32Array(frames)
  for (let i = 0; i < frames; i++) {
    const at = dataStart + i * bytesPerSample * channels
    let value = 0
    if (format === 3 && bits === 32) value = buf.readFloatLE(at)
    else if (bits === 16) value = buf.readInt16LE(at) / 32768
    else if (bits === 8) value = (buf.readUInt8(at) - 128) / 128
    else if (bits === 32) value = buf.readInt32LE(at) / 2147483648
    else if (bits === 24) {
      const raw = buf.readUIntLE(at, 3)
      value = (raw & 0x800000 ? raw - 0x1000000 : raw) / 8388608
    }
    out[i] = value
  }
  return { sampleRate, samples: out }
}

function loadSherpa(file) {
  if (!file) return null
  if (sherpaCache.file === file && sherpaCache.mod) return sherpaCache.mod
  const req = createRequire(import.meta.url)
  const mod = req(file)
  sherpaCache = { file, mod }
  return mod
}

export class VoiceTranscriber {
  constructor({ tools = {}, logger = null } = {}) {
    this.tools = { ...DEFAULT_VOICE_TOOLS, ...tools }
    this.logger = logger
    // 懒加载：第一次真正要转写时才加载模型（约 2.5–3 秒、约 300MB 内存）。
    this.engine = null
    this.engineError = ''
  }

  /** 把路径全部重新解析一遍：模型/可执行文件可能被移动或刚装好。 */
  resolvedTools() {
    const t = this.tools
    const modelDir = SENSEVOICE_MODEL_DIRS.find((dir) => {
      try { return fs.statSync(path.join(dir, 'tokens.txt')).isFile() } catch { return false }
    }) ?? ''
    const senseVoiceModel = firstExisting([
      t.senseVoiceModel,
      modelDir ? path.join(modelDir, 'model.int8.onnx') : '',
      modelDir ? path.join(modelDir, 'model.onnx') : '',
    ], '.onnx')
    const senseVoiceTokens = firstExisting([
      t.senseVoiceTokens,
      modelDir ? path.join(modelDir, 'tokens.txt') : '',
    ], '.txt')
    const sileroVad = firstExisting([t.sileroVad, ...SILERO_MODEL_CANDIDATES], '.onnx')
    const sherpaNative = firstExisting([t.sherpaNative, ...SHERPA_NATIVE_CANDIDATES], '.node')
    return {
      silkDecoder: t.silkDecoder,
      ffmpeg: t.ffmpeg,
      senseVoiceModel,
      senseVoiceTokens,
      sileroVad,
      sherpaNative,
    }
  }

  describe() {
    const resolved = this.resolvedTools()
    const rows = [
      ['silkDecoder', resolved.silkDecoder],
      ['ffmpeg', resolved.ffmpeg],
      ['senseVoiceModel', resolved.senseVoiceModel],
      ['senseVoiceTokens', resolved.senseVoiceTokens],
      ['sherpaNative', resolved.sherpaNative],
    ]
    const missing = rows.filter(([, file]) => {
      try { return !fs.statSync(file).isFile() } catch { return true }
    }).map(([name, file]) => ({ name, file }))
    return {
      // 引擎是 SenseVoice；VAD 有就用（分段更准），没有也不影响出字。
      engine: 'sensevoice',
      ok: missing.length === 0,
      missing,
      vad: resolved.sileroVad || '',
      tools: resolved,
    }
  }

  /** 懒加载 SenseVoice + VAD；失败时把原因留在 engineError 里。 */
  engineOrThrow() {
    if (this.engine) return this.engine
    const resolved = this.resolvedTools()
    if (!resolved.sherpaNative) throw new Error('找不到 sherpa-onnx 原生模块（sherpa-onnx.node）')
    const sherpa = loadSherpa(resolved.sherpaNative)
    if (!sherpa) throw new Error('sherpa-onnx 原生模块加载失败')
    const started = Date.now()
    const recognizer = sherpa.createOfflineRecognizer({
      featConfig: { sampleRate: 16000, featureDim: 80 },
      modelConfig: {
        senseVoice: {
          model: resolved.senseVoiceModel,
          language: 'auto',
          useInverseTextNormalization: 1,
        },
        tokens: resolved.senseVoiceTokens,
        numThreads: Math.min(4, os.cpus()?.length ?? 4),
        provider: 'cpu',
        debug: 0,
      },
    })
    // 不加载 Silero VAD：它的 voiceActivityDetectorFront() 在 Electron 下必抛
    // External buffers are not allowed（见 recognizeWav 注释），加载了也用不上。
    const vad = null
    this.engine = { sherpa, recognizer, vad, loadMs: Date.now() - started }
    this.logger?.info?.('[qq-console-v2] SenseVoice 已加载（%dms）', this.engine.loadMs)
    return this.engine
  }

  /**
   * 16k 单声道 wav → 文本。
   *
   * 不用 Silero VAD 切段：`voiceActivityDetectorFront()` 会把段样本放进**原生层
   * 分配的外部 ArrayBuffer**，Electron（Node 24）下直接抛
   * `External buffers are not allowed`，实测必炸；而同一段音频走
   * `acceptWaveformOffline` 整段识别完全正常。所以这里改成定长窗口：
   * SenseVoice 一次能稳定吃下约 30 秒，短语音（QQ 语音大多 2–20 秒）直接整段识别。
   */
  recognizeWav(wavFile) {
    const { sherpa, recognizer } = this.engineOrThrow()
    const wave = readWavSamples(wavFile)
    const samples = wave?.samples
    if (!samples || !samples.length) return ''
    const decode = (segment) => {
      if (!segment || !segment.length) return ''
      // 一律复制成普通 Float32Array：即使来源是原生侧数组，喂回去也不会触发
      // External buffers 限制。
      const owned = Float32Array.from(segment)
      if (!owned.length) return ''
      const stream = sherpa.createOfflineStream(recognizer)
      sherpa.acceptWaveformOffline(stream, { samples: owned, sampleRate: 16000 })
      sherpa.decodeOfflineStream(recognizer, stream)
      const raw = sherpa.getOfflineStreamResultAsJson(stream)
      let text = ''
      try { text = JSON.parse(raw)?.text ?? '' } catch { text = '' }
      return cleanSenseVoiceText(text)
    }
    const windowSize = MAX_WHOLE_SAMPLES
    if (samples.length <= windowSize) return decode(samples)
    const texts = []
    for (let i = 0; i < samples.length; i += windowSize) {
      const text = decode(samples.subarray(i, Math.min(i + windowSize, samples.length)))
      if (text) texts.push(text)
    }
    return texts.join(' ')
  }

  async transcribe(filePath, { timeoutMs = 120_000 } = {}) {
    const file = path.resolve(String(filePath ?? ''))
    let stat
    try {
      stat = fs.statSync(file)
    } catch (err) {
      return { ok: false, error: 'file-not-found', detail: String(err?.message ?? err) }
    }
    if (!stat.isFile() || stat.size <= 0) {
      return { ok: false, error: 'empty-file' }
    }

    const availability = this.describe()
    if (!availability.ok) {
      return {
        ok: false,
        error: 'missing-tools',
        missing: availability.missing,
        detail: `缺少 ${availability.missing.map((x) => path.basename(x.file)).join('、')}`,
      }
    }

    const tmp = path.resolve(fs.mkdtempSync(path.join(os.tmpdir(), 'dsh-qq-voice-')))
    const wav = path.join(tmp, 'audio.wav')
    try {
      const head = readHead(file)
      const silkAt = head.indexOf(SILK_MAGIC)
      let method = 'audio'

      if (silkAt >= 0) {
        method = 'silk'
        const raw = fs.readFileSync(file)
        const silk = path.join(tmp, 'audio.silk')
        fs.writeFileSync(silk, raw.subarray(raw.indexOf(SILK_MAGIC)))
        const pcm = path.join(tmp, 'audio.pcm')
        const decoded = await run(availability.tools.silkDecoder, [silk, pcm], { timeoutMs: 60_000 })
        if (decoded.code !== 0 || !fs.existsSync(pcm)) {
          return {
            ok: false,
            error: 'silk-decode-failed',
            detail: String(decoded.stderr ?? '').trim().slice(-400),
          }
        }
        const converted = await run(availability.tools.ffmpeg, [
          '-hide_banner', '-loglevel', 'error', '-y',
          '-f', 's16le', '-ar', '24000', '-ac', '1', '-i', pcm,
          '-ar', '16000', '-ac', '1', '-c:a', 'pcm_s16le', wav,
        ], { timeoutMs: 60_000 })
        if (converted.code !== 0 || !fs.existsSync(wav)) {
          return {
            ok: false,
            error: 'silk-convert-failed',
            detail: String(converted.stderr ?? '').trim().slice(-400),
          }
        }
      } else {
        const converted = await run(availability.tools.ffmpeg, [
          '-hide_banner', '-loglevel', 'error', '-y',
          '-i', file, '-vn', '-ar', '16000', '-ac', '1', '-c:a', 'pcm_s16le', wav,
        ], { timeoutMs: 60_000 })
        if (converted.code !== 0 || !fs.existsSync(wav)) {
          return {
            ok: false,
            error: 'audio-convert-failed',
            detail: String(converted.stderr ?? '').trim().slice(-400),
          }
        }
      }

      let text
      try {
        text = this.recognizeWav(wav)
      } catch (err) {
        this.engineError = String(err?.message ?? err)
        return { ok: false, error: 'sensevoice-failed', detail: this.engineError }
      }
      text = String(text ?? '').replace(/\s+/g, ' ').trim()
      if (!validText(text)) return { ok: false, error: 'empty-or-noise' }
      this.logger?.info?.('[qq-console-v2] 语音转写完成（sensevoice-%s，%d 字）', method, text.length)
      return { ok: true, text, method: `sensevoice-${method}` }
    } catch (err) {
      return { ok: false, error: 'transcribe-exception', detail: String(err?.message ?? err) }
    } finally {
      // 只删除本函数用 mkdtemp 创建、且确实位于系统临时目录下的目录。
      try {
        const tempRoot = path.resolve(os.tmpdir())
        const prefix = tempRoot.endsWith(path.sep) ? tempRoot : `${tempRoot}${path.sep}`
        if (tmp.startsWith(prefix) && path.basename(tmp).startsWith('dsh-qq-voice-')) {
          fs.rmSync(tmp, { recursive: true, force: true })
        }
      } catch { /* 临时目录清理失败不影响转写结果 */ }
    }
  }
}
