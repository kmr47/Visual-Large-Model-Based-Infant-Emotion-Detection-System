<script setup>
import { computed, onMounted, onUnmounted, ref } from 'vue'
import {
  Activity,
  AlertTriangle,
  BarChart3,
  Bell,
  Camera,
  CheckCircle2,
  Clock3,
  Cloud,
  Pause,
  Play,
  StepForward,
  Database,
  Gauge,
  HeartPulse,
  Loader2,
  Radio,
  RefreshCw,
  Upload,
  Users,
  Video,
  Wifi,
  WifiOff,
} from '@lucide/vue'
import ReportView from './components/ReportView.vue'
import {
  emotionLabels,
  fallbackData,
  fetchAlerts,
  fetchCameras,
  fetchHealth,
  fetchStats,
  inferImage,
  inferVideo,
  negativeEmotions,
  openStream,
} from './services/api'

const loading = ref(false)
const backendOnline = ref(false)
const activeView = ref('dashboard') // dashboard | reports
const streamStatus = ref('closed')
const health = ref(fallbackData.health)
const cameras = ref(fallbackData.cameras)
const alerts = ref(fallbackData.alerts)
const stats = ref(fallbackData.stats)
const latestResult = ref(null)
const videoResult = ref(null)
const uploadError = ref('')
const selectedImageUrl = ref('')
const selectedVideoUrl = ref('')
const videoCurrentTime = ref(0)
const videoNaturalWidth = ref(360)
const videoNaturalHeight = ref(240)
const eventLog = ref([
  { type: 'system', text: '工作台已进入演示模式', time: new Date().toLocaleTimeString('zh-CN', { hour12: false }) },
])
const demoRunning = ref(false)
const demoFrame = ref(0)
const canvasRef = ref(null)
const videoRef = ref(null)
const videoContainerRef = ref(null)
const animationFrameId = ref(null)
let ws = null
let demoTimer = null

const emotionPalette = {
  anger: '#dd5f55',
  contempt: '#9b7a45',
  disgust: '#5c9a62',
  fear: '#8a6fd1',
  happy: '#e8b84b',
  neutral: '#5e8796',
  sad: '#5275b5',
  surprise: '#db7b43',
}

const orderedEmotionCounts = computed(() => {
  const counts = stats.value?.emotion_counts || {}
  return Object.keys(emotionLabels).map((key) => ({
    key,
    label: emotionLabels[key],
    count: Number(counts[key] || 0),
    color: emotionPalette[key],
  }))
})

const totalEmotionEvents = computed(() => orderedEmotionCounts.value.reduce((sum, item) => sum + item.count, 0))
const imageDetections = computed(() => {
  if (selectedImageUrl.value && latestResult.value?.detections?.length) {
    return latestResult.value.detections
  }
  return []
})
const activeAlerts = computed(() => alerts.value.filter((a) => a.status === 'active'))
const currentVideoFrame = computed(() => {
  const frames = videoResult.value?.frames || []
  if (!frames.length) return null
  return frames.reduce((best, frame) => {
    return Math.abs(frame.timestamp_sec - videoCurrentTime.value) < Math.abs(best.timestamp_sec - videoCurrentTime.value) ? frame : best
  }, frames[0])
})
const videoDetection = computed(() => currentVideoFrame.value?.detections?.[0] || null)
const latestDetection = computed(() => videoDetection.value || latestResult.value?.detections?.[0] || null)
const videoAggregate = computed(() => videoResult.value?.aggregate || null)
const negativeMoments = computed(() => {
  const frames = videoResult.value?.frames || []
  const moments = []
  let lastKey = ''
  let lastTs = -999
  for (const frame of frames) {
    const dets = frame.detections || []
    if (!dets.length) continue
    const negativeDets = dets.filter((d) => negativeEmotions.has(d.primary_emotion))
    if (!negativeDets.length) continue
    negativeDets.sort((a, b) => (b.negative_score ?? b.primary_confidence ?? 0) - (a.negative_score ?? a.primary_confidence ?? 0))
    const det = negativeDets[0]
    const key = `${det.track_id}-${det.primary_emotion}`
    const ts = Number(frame.timestamp_sec || 0)
    if (key === lastKey && ts - lastTs < 2.0) continue
    moments.push({
      timestamp: ts,
      frameIndex: frame.frame_index,
      trackId: det.track_id,
      emotion: det.primary_emotion,
      confidence: det.primary_confidence ?? 0,
      negativeScore: det.negative_score ?? 0,
    })
    lastKey = key
    lastTs = ts
    if (moments.length >= 12) break
  }
  return moments
})

function jumpToMoment(moment) {
  const video = videoRef.value
  if (!video || !moment) return
  video.currentTime = Math.max(0, Number(moment.timestamp || 0))
  videoCurrentTime.value = video.currentTime
  startCanvasAnimation()
}

const currentFrameDets = computed(() => {
  if (selectedVideoUrl.value) return currentVideoFrame.value?.detections || []
  return latestResult.value?.detections || []
})

const faceCount = computed(() => currentFrameDets.value.length)

const negativeScore = computed(() => {
  const dets = currentFrameDets.value
  if (!dets.length) return 0
  return Math.max(...dets.map((d) => d.negative_score ?? 0))
})

const primaryEmotion = computed(() => {
  const dets = currentFrameDets.value
  if (!dets.length) return 'neutral'
  // Return the most negative emotion (most concerning)
  const sorted = [...dets].sort((a, b) => (b.negative_score ?? 0) - (a.negative_score ?? 0))
  return sorted[0].primary_emotion || 'neutral'
})
const primaryLabel = computed(() => emotionLabels[primaryEmotion.value] || primaryEmotion.value)
const modelMode = computed(() => health.value?.inference_mode || 'unknown')

function faceBoxStyle(det) {
  const meta = latestResult.value?.meta || {}
  const width = Number(meta.image_width || meta.image_size?.[0] || 360)
  const height = Number(meta.image_height || meta.image_size?.[1] || 240)
  const bbox = det?.bbox || [90, 34, 180, 164]
  if (!width || !height) return {}
  return {
    left: `${bbox[0] / width * 100}%`,
    top: `${bbox[1] / height * 100}%`,
    width: `${bbox[2] / width * 100}%`,
    height: `${bbox[3] / height * 100}%`,
    borderColor: emotionPalette[det.primary_emotion],
  }
}

function emotionColor(emotion) {
  return emotionPalette[emotion] || '#e8b84b'
}

function displayId(det) {
  // Operators see the privacy pseudonym; the internal person-N track stays hidden.
  if (det?.anonymous_id) return det.anonymous_id
  return det?.track_id || '--'
}

function addLog(type, text) {
  eventLog.value.unshift({ type, text, time: new Date().toLocaleTimeString('zh-CN', { hour12: false }) })
  eventLog.value = eventLog.value.slice(0, 8)
}

async function refreshAll() {
  loading.value = true
  try {
    const [h, c, a, s] = await Promise.all([fetchHealth(), fetchCameras(), fetchAlerts(), fetchStats()])
    health.value = h
    cameras.value = c
    alerts.value = a.length ? a : fallbackData.alerts
    stats.value = s
    backendOnline.value = true
    addLog('system', `后端同步完成，模式 ${h.inference_mode}`)
  } catch (error) {
    backendOnline.value = false
    health.value = fallbackData.health
    cameras.value = fallbackData.cameras
    alerts.value = fallbackData.alerts
    stats.value = fallbackData.stats
    addLog('warn', '后端不可用，已使用本地演示数据')
  } finally {
    loading.value = false
  }
}

function connectStream() {
  if (ws) ws.close()
  try {
    ws = openStream((message) => {
      if (message.type === 'inference') {
        latestResult.value = message.result
        addLog('inference', `WebSocket 收到 ${message.result.detections?.[0]?.primary_emotion || 'unknown'} 结果`)
      }
      if (message.type === 'alert_created') {
        alerts.value.unshift(message.alert)
        addLog('alert', message.alert.message)
      }
    }, (status) => {
      streamStatus.value = status
    })
  } catch {
    streamStatus.value = 'error'
  }
}

async function submitImage(file, previewUrl = '') {
  uploadError.value = ''
  stopCanvasAnimation()
  // 清空旧框防止滞留
  latestResult.value = null
  if (previewUrl) selectedImageUrl.value = previewUrl
  loading.value = true
  try {
    const result = await inferImage(file)
    videoResult.value = null
    selectedVideoUrl.value = ''
    videoCurrentTime.value = 0
    videoNaturalWidth.value = 360
    videoNaturalHeight.value = 240
    latestResult.value = result
    if (result.alerts?.length) {
      alerts.value = [...result.alerts, ...alerts.value]
      addLog('alert', result.alerts[0].message)
    } else {
      addLog('inference', `图片推理完成：${result.detections?.[0]?.primary_emotion || 'no face'}`)
    }
    await refreshStatsOnly()
  } catch (error) {
    uploadError.value = error.message || '上传推理失败'
    addLog('warn', '图片推理失败')
    stopDemo()
  } finally {
    loading.value = false
  }
}

async function handleUpload(event) {
  const file = event.target.files?.[0]
  if (!file) return
  stopDemo()
  await submitImage(file, URL.createObjectURL(file))
  event.target.value = ''
}

async function handleVideoUpload(event) {
  const file = event.target.files?.[0]
  if (!file) return
  uploadError.value = ''
  stopDemo()
  stopCanvasAnimation()
  // Clear stale video/image overlays before a new upload. Otherwise failed/slow uploads can
  // keep drawing boxes from the previous analysis and make backend fixes look ineffective.
  videoResult.value = null
  latestResult.value = null
  selectedImageUrl.value = ''
  selectedVideoUrl.value = ''
  videoCurrentTime.value = 0
  const ctx = canvasRef.value?.getContext?.('2d')
  if (ctx) ctx.clearRect(0, 0, canvasRef.value.width, canvasRef.value.height)
  loading.value = true
  try {
    const result = await inferVideo(file)
    videoResult.value = result
    latestResult.value = null
    selectedImageUrl.value = ''
    selectedVideoUrl.value = URL.createObjectURL(file)
    videoCurrentTime.value = 0
    const firstWithFace = result.frames?.find((f) => f.detections?.length)
    const debug = result.aggregate?.smoothing?.bbox_source || result.aggregate?.bbox_source || result.aggregate?.debug || ''
    addLog('inference', `视频分析完成：${result.frames.length} 帧，首个人脸帧=${firstWithFace?.frame_index ?? '无'} ${debug}`)
  } catch (error) {
    uploadError.value = error.message || '视频推理失败'
    addLog('warn', '视频推理失败')
  } finally {
    loading.value = false
    event.target.value = ''
  }
}

function handleVideoLoaded(event) {
  videoNaturalWidth.value = event.target.videoWidth || 360
  videoNaturalHeight.value = event.target.videoHeight || 240
}

function handleVideoTimeUpdate(event) {
  videoCurrentTime.value = event.target.currentTime || 0
}

function startCanvasAnimation() {
  stopCanvasAnimation()
  animationFrameId.value = requestAnimationFrame(drawCanvasLoop)
}

function stopCanvasAnimation() {
  if (animationFrameId.value) {
    cancelAnimationFrame(animationFrameId.value)
    animationFrameId.value = null
  }
}

function drawCanvasLoop() {
  const canvas = canvasRef.value
  const video = videoRef.value
  if (!canvas || !video) {
    animationFrameId.value = requestAnimationFrame(drawCanvasLoop)
    return
  }

  const cw = canvas.parentElement?.clientWidth || 360
  const ch = canvas.parentElement?.clientHeight || 240
  if (cw < 10 || ch < 10) {
    animationFrameId.value = requestAnimationFrame(drawCanvasLoop)
    return
  }

  if (canvas.width !== cw || canvas.height !== ch) {
    canvas.width = cw
    canvas.height = ch
  }

  const ctx = canvas.getContext('2d')
  ctx.clearRect(0, 0, canvas.width, canvas.height)

  // Sync current video time
  videoCurrentTime.value = video.currentTime

  const curFrame = currentVideoFrame.value
  const dets = curFrame?.detections || []
  if (!dets.length || videoNaturalWidth.value < 10) {
    animationFrameId.value = requestAnimationFrame(drawCanvasLoop)
    return
  }

  // Calculate displayed video region (object-fit: contain)
  const vA = videoNaturalWidth.value / videoNaturalHeight.value
  const cA = cw / ch
  let displayW, displayH, ox, oy
  if (vA > cA) {
    displayW = cw
    displayH = cw / vA
    ox = 0
    oy = (ch - displayH) / 2
  } else {
    displayH = ch
    displayW = ch * vA
    ox = (cw - displayW) / 2
    oy = 0
  }

  const scX = displayW / videoNaturalWidth.value
  const scY = displayH / videoNaturalHeight.value
  const lw = Math.max(2, Math.min(4, canvas.width / 180))
  const labelFs = Math.max(13, Math.min(18, canvas.width / 26))

  // --- Draw each face ---
  dets.forEach((det) => {
    if (!det.bbox) return
    const [bx, by, bw, bh] = det.bbox
    const x = ox + bx * scX
    const y = oy + by * scY
    const w = bw * scX
    const h = bh * scY
    const color = emotionPalette[det.primary_emotion] || '#00ff88'

    // Face rectangle with glow
    ctx.strokeStyle = color
    ctx.lineWidth = lw
    ctx.shadowColor = color
    ctx.shadowBlur = 12
    ctx.strokeRect(x, y, w, h)
    ctx.shadowBlur = 0

    // Corner markers
    const cl = Math.min(20, Math.max(10, w / 6))
    ctx.lineWidth = lw * 1.2
    ;[
      [[x, y], [x + cl, y]], [[x, y], [x, y + cl]],
      [[x + w, y], [x + w - cl, y]], [[x + w, y], [x + w, y + cl]],
      [[x, y + h], [x + cl, y + h]], [[x, y + h], [x, y + h - cl]],
      [[x + w, y + h], [x + w - cl, y + h]], [[x + w, y + h], [x + w, y + h - cl]],
    ].forEach(([from, to]) => {
      ctx.beginPath()
      ctx.moveTo(from[0], from[1])
      ctx.lineTo(to[0], to[1])
      ctx.strokeStyle = color
      ctx.stroke()
    })

    // Label above face
    const label = `${emotionLabels[det.primary_emotion] || det.primary_emotion} ${Math.round(det.primary_confidence * 100)}%`
    ctx.font = `bold ${labelFs}px sans-serif`
    const textW = ctx.measureText(label).width
    const lPad = 8
    const lH = labelFs + lPad * 2
    const labelX = Math.max(0, Math.min(x, canvas.width - textW - lPad * 2 - 4))
    const labelY = Math.max(0, y - lH - 4)

    ctx.fillStyle = 'rgba(0,0,0,0.78)'
    ctx.fillRect(labelX, labelY, textW + lPad * 2 + 4, lH)
    ctx.fillStyle = color
    ctx.fillText(label, labelX + lPad + 2, labelY + lH - lPad - 2)
  })

  // --- Top HUD bar: show face count + most concerning emotion ---
  const maxNeg = Math.max(...dets.map((d) => d.negative_score ?? 0))
  const worst = dets.reduce((a, b) => ((b.negative_score ?? 0) > (a.negative_score ?? 0) ? b : a), dets[0])
  const hudText = `${dets.length} \u4eba \u00b7 ${emotionLabels[worst.primary_emotion] || worst.primary_emotion} ${Math.round(worst.primary_confidence * 100)}% \u00b7 \u8d1f\u9762 ${Math.round(maxNeg * 100)}%`
  const hudFs = Math.max(12, Math.min(15, canvas.width / 32))
  ctx.font = `bold ${hudFs}px sans-serif`
  const hudW = ctx.measureText(hudText).width + 24
  const hudH = 34
  const hudX = 10
  const hudY = 10

  ctx.fillStyle = 'rgba(0,0,0,0.65)'
  ctx.beginPath()
  ctx.roundRect(hudX, hudY, hudW + 16, hudH, 6)
  ctx.fill()
  ctx.fillStyle = '#ffffff'
  ctx.fillText(hudText, hudX + 16, hudY + hudH - 10)

  // --- Bottom tracking indicator ---
  const btmText = `frame ${curFrame?.frame_index || '--'} \u00b7 ${dets.length} faces`
  const btmFs = Math.max(10, Math.min(12, canvas.width / 36))
  ctx.font = `${btmFs}px sans-serif`
  const btmW = ctx.measureText(btmText).width + 14
  ctx.fillStyle = 'rgba(0,0,0,0.55)'
  ctx.beginPath()
  ctx.roundRect(canvas.width - btmW - 10, canvas.height - 30, btmW, 22, 4)
  ctx.fill()
  ctx.fillStyle = '#aaa'
  ctx.fillText(btmText, canvas.width - btmW - 4, canvas.height - 12)

  animationFrameId.value = requestAnimationFrame(drawCanvasLoop)
}

async function refreshStatsOnly() {
  try {
    const [a, s] = await Promise.all([fetchAlerts(), fetchStats()])
    alerts.value = a.length ? a : alerts.value
    stats.value = s
  } catch {
    // Keep current UI state.
  }
}

function formatTime(value) {
  if (!value) return '--'
  return new Date(value).toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit', hour12: false })
}

function percent(value) {
  return `${Math.round((value || 0) * 100)}%`
}

function buildDemoSvg(frameIndex) {
  const palette = ['#dae8df', '#f4d17b', '#a7c7b5', '#d88c72', '#97a9cc']
  const bg = palette[frameIndex % palette.length]
  const accent = palette[(frameIndex + 2) % palette.length]
  const eyeY = 102 + (frameIndex % 3) * 2
  const mouth = frameIndex % 4 === 0 ? 'M114 148 Q150 128 186 148' : frameIndex % 4 === 1 ? 'M116 144 Q150 164 184 144' : 'M118 150 L184 150'
  return `
    <svg xmlns="http://www.w3.org/2000/svg" width="360" height="240" viewBox="0 0 360 240">
      <rect width="360" height="240" fill="${bg}"/>
      <rect x="22" y="22" width="316" height="196" rx="18" fill="rgba(255,255,255,.42)"/>
      <circle cx="150" cy="118" r="72" fill="#f2c7a5"/>
      <path d="M94 118 Q150 46 206 118" fill="${accent}" opacity=".72"/>
      <circle cx="124" cy="${eyeY}" r="8" fill="#1d2926"/>
      <circle cx="176" cy="${eyeY}" r="8" fill="#1d2926"/>
      <path d="${mouth}" stroke="#1d2926" stroke-width="7" fill="none" stroke-linecap="round"/>
      <text x="24" y="214" font-family="Arial" font-size="14" fill="#27443b">demo frame ${frameIndex + 1}</text>
    </svg>`
}

function makeDemoFile(frameIndex) {
  const svg = buildDemoSvg(frameIndex)
  const blob = new Blob([svg], { type: 'image/svg+xml' })
  return new File([blob], `demo-frame-${String(frameIndex + 1).padStart(3, '0')}.svg`, { type: 'image/svg+xml' })
}

async function runDemoStep() {
  const file = makeDemoFile(demoFrame.value)
  const previewUrl = URL.createObjectURL(file)
  demoFrame.value += 1
  await submitImage(file, previewUrl)
}

async function startDemo() {
  if (demoRunning.value) return
  demoRunning.value = true
  addLog('system', '演示播放已启动')
  await runDemoStep()
  demoTimer = window.setInterval(() => {
    if (!loading.value) runDemoStep()
  }, 2200)
}

function stopDemo() {
  demoRunning.value = false
  if (demoTimer) {
    window.clearInterval(demoTimer)
    demoTimer = null
  }
}

async function toggleDemo() {
  if (demoRunning.value) {
    stopDemo()
    addLog('system', '演示播放已暂停')
  } else {
    await startDemo()
  }
}

onMounted(async () => {
  await refreshAll()
  connectStream()
})

onUnmounted(() => {
  stopDemo()
  stopCanvasAnimation()
  if (ws) ws.close()
})
</script>

<template>
  <main class="dashboard-shell">
    <aside class="rail">
      <div class="brand-mark">
        <HeartPulse :size="24" />
      </div>
      <button
        class="rail-button"
        :class="{ active: activeView === 'dashboard' }"
        title="实时工作台"
        @click="activeView = 'dashboard'"
      ><Gauge :size="20" /></button>
      <button class="rail-button" title="摄像头"><Camera :size="20" /></button>
      <button class="rail-button" title="告警"><Bell :size="20" /></button>
      <button class="rail-button" title="数据"><Database :size="20" /></button>
      <button
        class="rail-button"
        :class="{ active: activeView === 'reports' }"
        title="情绪报表"
        @click="activeView = 'reports'"
      ><BarChart3 :size="20" /></button>
    </aside>

    <ReportView
      v-if="activeView === 'reports'"
      class="report-view"
      :backend-online="backendOnline"
      @back="activeView = 'dashboard'"
    />

    <section v-else class="workspace">
      <header class="topbar">
        <div>
          <p class="eyebrow">Child Emotion Operations</p>
          <h1>幼儿情绪检测工作台</h1>
        </div>
        <div class="top-actions">
          <span class="status-pill" :class="backendOnline ? 'online' : 'offline'">
            <Wifi v-if="backendOnline" :size="16" />
            <WifiOff v-else :size="16" />
            {{ backendOnline ? '后端在线' : '演示数据' }}
          </span>
          <span class="status-pill muted">
            <Cloud :size="16" />
            {{ modelMode }}
          </span>
          <button class="icon-button" @click="refreshAll" title="刷新数据">
            <Loader2 v-if="loading" class="spin" :size="18" />
            <RefreshCw v-else :size="18" />
          </button>
        </div>
      </header>

      <section class="metrics-grid">
        <article class="metric-panel">
          <div class="metric-icon calm"><Camera :size="19" /></div>
          <div>
            <span>摄像头</span>
            <strong>{{ cameras.length }}</strong>
          </div>
        </article>
        <article class="metric-panel">
          <div class="metric-icon warn"><AlertTriangle :size="19" /></div>
          <div>
            <span>活跃告警</span>
            <strong>{{ activeAlerts.length }}</strong>
          </div>
        </article>
        <article class="metric-panel">
          <div class="metric-icon pulse"><Users :size="19" /></div>
          <div>
            <span>监测到人脸</span>
            <strong>{{ faceCount }}</strong>
          </div>
        </article>
        <article class="metric-panel">
          <div class="metric-icon ok"><CheckCircle2 :size="19" /></div>
          <div>
            <span>负面得分(最高)</span>
            <strong>{{ percent(negativeScore) }}</strong>
          </div>
        </article>
      </section>

      <section class="main-grid">
        <article class="monitor-panel">
          <div class="panel-head">
            <div>
              <p class="eyebrow">Live Monitor</p>
              <h2>实时画面</h2>
            </div>
            <div class="monitor-actions">
              <button class="demo-button" @click="toggleDemo">
                <Pause v-if="demoRunning" :size="17" />
                <Play v-else :size="17" />
                {{ demoRunning ? '暂停演示' : '演示播放' }}
              </button>
              <button class="step-button" :disabled="loading" @click="runDemoStep" title="单步推理">
                <StepForward :size="17" />
              </button>
              <label class="upload-button">
                <Upload :size="17" />
                上传图片
                <input type="file" accept="image/*" @change="handleUpload" />
              </label>
              <label class="upload-button video-upload">
                <Video :size="17" />
                上传视频
                <input type="file" accept="video/*" @change="handleVideoUpload" />
              </label>
            </div>
          </div>

          <div class="video-surface">
            <img v-if="selectedImageUrl" :src="selectedImageUrl" alt="上传图片预览" />
            <div v-else-if="selectedVideoUrl" class="video-overlay-container" ref="videoContainerRef">
              <video
                ref="videoRef"
                class="video-player"
                :src="selectedVideoUrl"
                controls
                autoplay
                muted
                @loadedmetadata="handleVideoLoaded"
                @timeupdate="handleVideoTimeUpdate"
                @play="startCanvasAnimation"
                @pause="stopCanvasAnimation"
                @ended="stopCanvasAnimation"
              ></video>
              <canvas ref="canvasRef" class="video-canvas"></canvas>
              <div class="stream-chip">
                <span></span>{{ `${videoCurrentTime.toFixed(1)}s` }}
              </div>
            </div>
            <div v-else class="camera-placeholder">
              <Radio :size="44" />
              <span>等待上传图片或实时流</span>
            </div>
            <div v-for="det in imageDetections" :key="det.track_id" class="face-box" :style="faceBoxStyle(det)">
              <span :style="{ background: emotionColor(det.primary_emotion) }">{{ emotionLabels[det.primary_emotion] || det.primary_emotion }} · {{ percent(det.primary_confidence) }}</span>
            </div>
            <div class="stream-chip" v-if="!selectedVideoUrl" :class="streamStatus">
              <span></span>{{ demoRunning ? `demo ${demoFrame}` : streamStatus }}
            </div>
          </div>
          <div class="video-progress" v-if="selectedVideoUrl && videoResult">
            <span>帧 {{ currentVideoFrame?.frame_index || '--' }} · {{ faceCount }} 人脸</span>
            <b>{{ primaryLabel }} {{ percent(negativeScore) }}</b>
            <i :style="{ width: `${Math.min(100, videoCurrentTime / Math.max(videoResult.duration_sec || 1, 1) * 100)}%` }"></i>
          </div>
          <div class="debug-panel" v-if="selectedVideoUrl && videoResult?.aggregate?.smoothing">
            <b>调试信息</b>
            <span>bbox: {{ videoResult.aggregate.smoothing.bbox_source || 'unknown' }}</span>
            <span>raw帧: {{ videoResult.aggregate.smoothing.raw_bbox_frames }}</span>
            <span>渲染帧: {{ videoResult.aggregate.smoothing.rendered_bbox_frames }}</span>
            <span>maxRaw: {{ videoResult.aggregate.smoothing.max_raw_bboxes }}</span>
            <span>maxShow: {{ videoResult.aggregate.smoothing.max_rendered_bboxes }}</span>
            <span>verifier: {{ videoResult.aggregate.smoothing.verifier || '-' }}</span>
            <span>emotion: {{ videoResult.aggregate.smoothing.emotion_source || '-' }}</span>
            <span>cloudCalls: {{ videoResult.aggregate.smoothing.cloud_crop_calls ?? '-' }}</span>
            <span>cloudErrors: {{ videoResult.aggregate.smoothing.cloud_crop_errors ?? '-' }}</span>
            <span>localCalls: {{ videoResult.aggregate.smoothing.local_crop_calls ?? '-' }}</span>
            <span>localErrors: {{ videoResult.aggregate.smoothing.local_crop_errors ?? '-' }}</span>
            <span v-if="videoResult.aggregate.smoothing.cloud_last_error">lastError: {{ videoResult.aggregate.smoothing.cloud_last_error }}</span>
          </div>
          <div class="demo-progress" v-if="demoFrame > 0 && !selectedVideoUrl">
            <span>演示帧</span>
            <b>{{ demoFrame }}</b>
            <i :style="{ width: `${Math.min(100, (demoFrame % 12) / 12 * 100)}%` }"></i>
          </div>
          <div class="negative-moments" v-if="selectedVideoUrl && negativeMoments.length">
            <div class="moments-head">
              <strong>负面片段</strong>
              <span>点击跳转验证</span>
            </div>
            <button
              v-for="moment in negativeMoments"
              :key="`${moment.frameIndex}-${moment.trackId}-${moment.emotion}`"
              type="button"
              class="moment-chip"
              :style="{ borderColor: emotionColor(moment.emotion), color: emotionColor(moment.emotion) }"
              @click="jumpToMoment(moment)"
            >
              {{ moment.timestamp.toFixed(1) }}s · {{ moment.trackId }} · {{ emotionLabels[moment.emotion] || moment.emotion }} {{ percent(moment.confidence) }}
            </button>
          </div>
          <div class="video-aggregate" v-if="videoResult?.aggregate?.emotion_distribution">
            <div class="panel-head compact" style="margin-top:14px">
              <div>
                <p class="eyebrow">Video Analysis</p>
                <h2>视频情绪分布</h2>
              </div>
            </div>
            <div class="distribution-list">
              <div v-for="(val, key) in videoResult.aggregate.emotion_distribution" :key="key" class="distribution-row">
                <span>{{ emotionLabels[key] || key }}</span>
                <div class="distribution-bar"><i :style="{ width: Math.max(4, val.pct) + '%', background: emotionPalette[key] }"></i></div>
                <b>{{ val.count }}</b>
              </div>
            </div>
          </div>
          <p class="bbox-note" v-if="faceCount > 0">
            {{ faceCount > 1 ? `${faceCount} 人同框检测，各人脸框与情绪标签独立显示。` : '单人脸检测。' }}演示框为后端返回的脸部区域估计。
          </p>
          <p v-if="uploadError" class="error-text">{{ uploadError }}</p>

          <div class="emotion-strip" v-if="currentFrameDets.length > 0">
            <div v-for="face in currentFrameDets" :key="face.track_id" class="face-card">
              <div class="face-card-header" :style="{ borderLeftColor: emotionColor(face.primary_emotion) }">
                <strong :title="face.track_id">{{ displayId(face) }}</strong>
                <span :style="{ color: emotionColor(face.primary_emotion) }">{{ emotionLabels[face.primary_emotion] }} · {{ percent(face.primary_confidence) }}</span>
              </div>
              <div v-for="item in face.emotions" :key="item.emotion" class="emotion-score">
                <div class="score-label">
                  <span>{{ item.label_cn || emotionLabels[item.emotion] }}</span>
                  <b>{{ percent(item.confidence) }}</b>
                </div>
                <div class="score-bar"><i :style="{ width: percent(item.confidence), background: emotionPalette[item.emotion] }"></i></div>
              </div>
            </div>
          </div>
        </article>

        <aside class="alerts-panel">
          <div class="panel-head compact">
            <div>
              <p class="eyebrow">Alert Queue</p>
              <h2>告警队列</h2>
            </div>
            <span class="count-badge">{{ alerts.length }}</span>
          </div>
          <div class="alert-list">
            <article v-for="alert in alerts.slice(0, 6)" :key="alert.id" class="alert-item" :class="alert.severity">
              <div class="alert-indicator"></div>
              <div class="alert-body">
                <div class="alert-title">
                  <strong>{{ emotionLabels[alert.emotion] || alert.emotion }}</strong>
                  <span>{{ formatTime(alert.created_at) }}</span>
                </div>
                <p>{{ alert.message }}</p>
                <div class="alert-meta">
                  <span>{{ alert.camera_id }}</span>
                  <span :title="alert.track_id">{{ displayId(alert) }}</span>
                  <span>{{ percent(alert.negative_score) }}</span>
                </div>
              </div>
            </article>
          </div>
        </aside>
      </section>

      <section class="lower-grid">
        <article class="distribution-panel">
          <div class="panel-head compact">
            <div>
              <p class="eyebrow">Distribution</p>
              <h2>情绪分布</h2>
            </div>
          </div>
          <div class="distribution-list">
            <div v-for="item in orderedEmotionCounts" :key="item.key" class="distribution-row">
              <span>{{ item.label }}</span>
              <div class="distribution-bar"><i :style="{ width: totalEmotionEvents ? `${Math.max(4, item.count / totalEmotionEvents * 100)}%` : '4%', background: item.color }"></i></div>
              <b>{{ item.count }}</b>
            </div>
          </div>
        </article>

        <article class="camera-panel">
          <div class="panel-head compact">
            <div>
              <p class="eyebrow">Devices</p>
              <h2>摄像头状态</h2>
            </div>
          </div>
          <div class="camera-list">
            <div v-for="camera in cameras" :key="camera.id" class="camera-row">
              <Camera :size="18" />
              <div>
                <strong>{{ camera.name }}</strong>
                <span>{{ camera.location || camera.source }}</span>
              </div>
              <em :class="camera.enabled ? 'enabled' : 'disabled'">{{ camera.enabled ? '运行中' : '停用' }}</em>
            </div>
          </div>
        </article>

        <article class="log-panel">
          <div class="panel-head compact">
            <div>
              <p class="eyebrow">Timeline</p>
              <h2>系统事件</h2>
            </div>
          </div>
          <div class="log-list">
            <div v-for="entry in eventLog" :key="entry.time + entry.text" class="log-row">
              <Clock3 :size="15" />
              <span>{{ entry.time }}</span>
              <p>{{ entry.text }}</p>
            </div>
          </div>
        </article>
      </section>
    </section>
  </main>
</template>
