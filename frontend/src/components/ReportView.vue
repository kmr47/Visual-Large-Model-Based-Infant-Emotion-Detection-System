<script setup>
import { computed, onMounted, ref, watch } from 'vue'
import {
  BarChart3,
  CalendarDays,
  CalendarRange,
  Database,
  Download,
  FlaskConical,
  KeyRound,
  Loader2,
  Lock,
  RefreshCw,
  Shield,
  Sparkles,
  Trash2,
  Users,
} from '@lucide/vue'
import {
  fetchChildDaily,
  fetchChildren,
  fetchDailyReport,
  fetchEventBatches,
  fetchPrivacyStatus,
  fetchWeeklyReport,
  purgeAllEvents,
  seedDemoEvents,
  createChild,
  deleteChild,
  claimTrack,
  unclaimTrack,
  fetchHealth,
  checkAdminToken,
  fetchIdentities,
  fetchPendingConfirmations,
  resolvePendingConfirmation,
  claimIdentity,
  unclaimIdentity,
  deleteIdentity,
  thumbUrl,
  burnTrackThumb,
  burnRequestThumbs,
} from '../services/api'

const props = defineProps({
  backendOnline: { type: Boolean, default: true },
})
const emit = defineEmits(['back'])

const viewMode = ref('daily') // daily | weekly
const loading = ref(false)
const error = ref('')
const daily = ref(null)
const weekly = ref(null)
const privacy = ref(null)
const adminToken = ref((() => {
  try { return sessionStorage.getItem('emotion-admin-token') || '' } catch { return '' }
})())
const authState = ref('idle') // idle | checking | ok | bad
let authTimer = null

async function checkAuth() {
  const token = adminToken.value.trim()
  if (!token) {
    authState.value = 'idle'
    try { sessionStorage.removeItem('emotion-admin-token') } catch {}
    return
  }
  authState.value = 'checking'
  try {
    const r = await checkAdminToken(token)
    authState.value = r.authorized ? 'ok' : 'bad'
    if (r.authorized) {
      try { sessionStorage.setItem('emotion-admin-token', token) } catch {}
    } else {
      try { sessionStorage.removeItem('emotion-admin-token') } catch {}
    }
  } catch {
    authState.value = 'bad'
  }
}

watch(adminToken, () => {
  clearTimeout(authTimer)
  authTimer = setTimeout(checkAuth, 400)
})
const adminOpen = ref(false)
const adminMessage = ref('')

// ---- child roster / per-child mode ----
const children = ref([])
const selectedChild = ref(null) // null = class/group aggregate
const childLoading = ref(false)
const rosterOpen = ref(false)
const rosterMessage = ref('')
const newChildName = ref('')
const newChildClass = ref('')

// ---- recent detection batches (pseudonym + claim) ----
const batches = ref([])
const batchesOpen = ref(false)
const batchesLoading = ref(false)
const identities = ref([])
const identitiesOpen = ref(false)
const identitiesLoading = ref(false)
const pendingList = ref([])
const identityMsg = ref('')
const identityClaimSel = ref({}) // anon_code -> child id or '__new__'
const claimMsg = ref('')
const claimSel = ref({}) // key: request|track -> child id or '__new__'
const claimNewName = ref('')

const EMOTION_ORDER = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
const LABELS = {
  anger: '生气', contempt: '轻蔑', disgust: '厌恶', fear: '害怕',
  happy: '开心', neutral: '平静', sad: '伤心', surprise: '惊讶',
}
const PALETTE = {
  anger: '#dd5f55', contempt: '#9b7a45', disgust: '#5c9a62', fear: '#8a6fd1',
  happy: '#e8b84b', neutral: '#5e8796', sad: '#5275b5', surprise: '#db7b43',
}
const NEGATIVE = new Set(['anger', 'disgust', 'fear', 'sad'])

function todayStr() {
  const d = new Date()
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`
}
function shiftDay(iso, delta) {
  const d = new Date(iso + 'T00:00:00')
  d.setDate(d.getDate() + delta)
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`
}
function shiftWeek(iso, delta) {
  return shiftDay(iso, delta * 7)
}

const date = ref(todayStr())
const weekStart = ref(todayStr())

const activeData = computed(() => (viewMode.value === 'daily' ? daily.value : weekly.value))
const activeLabel = computed(() => activeData.value?.label_cn || '')
const viewingChild = computed(() => children.value.find((c) => c.id === selectedChild.value) || null)
const isPerChild = computed(() => !!selectedChild.value)

const emotionRows = computed(() => {
  const data = activeData.value
  if (!data) return []
  const counts = data.emotion_counts || {}
  const total = data.total_events || 1
  return EMOTION_ORDER.map((key) => ({
    key,
    label: LABELS[key],
    count: counts[key] || 0,
    pct: Math.round(((counts[key] || 0) / total) * 1000) / 10,
    color: PALETTE[key],
    negative: NEGATIVE.has(key),
  })).sort((a, b) => b.count - a.count)
})

const totalEvents = computed(() => activeData.value?.total_events || 0)
const negativeEvents = computed(() => activeData.value?.negative_events || 0)
const negativeRatio = computed(() => {
  const d = activeData.value
  if (!d || !d.total_events) return 0
  return Math.round((d.negative_events / d.total_events) * 1000) / 10
})
const distinctChildren = computed(() => activeData.value?.distinct_children ?? 0)
const alertCount = computed(() => activeData.value?.alert_count || 0)
const mainEmotion = computed(() => {
  const d = activeData.value
  if (!d) return '--'
  const key = d.summary?.main_emotion || (emotionRows.value[0]?.key)
  return LABELS[key] || key || '--'
})
const mainEmotionColor = computed(() => PALETTE[activeData.value?.summary?.main_emotion] || '#888')

const hourlyRows = computed(() => {
  const dist = activeData.value?.hourly_distribution
  if (!dist || !dist.length) return []
  const max = Math.max(...dist, 1)
  return dist.map((count, hour) => ({
    hour,
    count,
    pct: Math.round((count / max) * 100),
  }))
})

const dailySeries = computed(() => {
  const rows = weekly.value?.daily_series || []
  const max = Math.max(1, ...rows.map((r) => r.total_events))
  return rows.map((r) => ({
    ...r,
    pct: Math.max(4, Math.round((r.total_events / max) * 100)),
    dayLabel: r.date.slice(5).replace('-', '/'),
  }))
})

function deltaText(vs, field) {
  if (!vs || vs[field] === undefined) return null
  const v = vs[field]
  if (v === 0) return { text: '持平', cls: 'flat' }
  return v > 0
    ? { text: `+${v}`, cls: 'up' }
    : { text: `${v}`, cls: 'down' }
}

function downloadJson(filename, obj) {
  const blob = new Blob([JSON.stringify(obj, null, 2)], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.click()
  URL.revokeObjectURL(url)
}

function emotionColor(emotion) {
  return PALETTE[emotion] || '#e8b84b'
}

async function load() {
  loading.value = true
  error.value = ''
  try {
    if (!children.value.length) {
      try {
        children.value = await fetchChildren()
      } catch { /* keep roster empty when backend down */ }
    }
    if (viewMode.value === 'daily') {
      daily.value = selectedChild.value
        ? await fetchChildDaily(selectedChild.value, date.value)
        : await fetchDailyReport(date.value)
    } else {
      weekly.value = await fetchWeeklyReport(weekStart.value)
    }
  } catch (e) {
    error.value = e.message || '报表加载失败'
  } finally {
    loading.value = false
  }
}

async function onChildChange() {
  // Re-fetch only when in daily (per-child weekly is a future extension).
  if (viewMode.value === 'daily') await load()
}

function switchMode(mode) {
  viewMode.value = mode
  if (mode === 'weekly') selectedChild.value = null // per-child weekly not implemented yet
  load()
}

function prevPeriod() {
  if (viewMode.value === 'daily') {
    date.value = shiftDay(date.value, -1)
  } else {
    weekStart.value = shiftWeek(weekStart.value, -1)
  }
  load()
}
function nextPeriod() {
  if (viewMode.value === 'daily') {
    date.value = shiftDay(date.value, +1)
  } else {
    weekStart.value = shiftWeek(weekStart.value, +1)
  }
  load()
}

async function adminAction(action) {
  adminMessage.value = ''
  const token = adminToken.value.trim() || undefined
  try {
    if (action === 'seed') {
      const res = await seedDemoEvents(token, { days: 7, perDay: 45 })
      adminMessage.value = `已生成 ${res.events} 条演示数据（source=demo）`
    } else if (action === 'purge') {
      const res = await purgeAllEvents(token)
      adminMessage.value = `已清除 ${res.emotion_events} 条事件 / ${res.alert_events} 条告警`
    }
    await load()
  } catch (e) {
    adminMessage.value = `失败: ${e.message || e}（后端若开启了访问控制，请填写管理口令）`
  }
}

function exportReport() {
  const data = activeData.value
  if (!data) return
  const scope = selectedChild.value ? `child-${selectedChild.value}` : 'group'
  const name = viewMode.value === 'daily' ? `daily-report-${scope}-${data.date}` : `weekly-report-${data.week_start}`
  downloadJson(`${name}.json`, data)
}

async function loadChildren() {
  try {
    children.value = await fetchChildren()
  } catch (e) {
    rosterMessage.value = `孩子档案加载失败: ${e.message || e}`
  }
}

async function addChild() {
  rosterMessage.value = ''
  const name = newChildName.value.trim()
  if (!name) {
    rosterMessage.value = '请填写昵称'
    return
  }
  try {
    const token = adminToken.value.trim() || undefined
    await createChild(token, { nickname: name, class_name: newChildClass.value.trim() })
    newChildName.value = ''
    newChildClass.value = ''
    rosterMessage.value = `已登记「${name}」`
    await loadChildren()
  } catch (e) {
    rosterMessage.value = `登记失败: ${e.message || e}（写操作需管理口令）`
  }
}

async function removeChild(childId, nickname) {
  if (!window.confirm(`删除档案「${nickname}」？其关联事件将自动匿名化。`)) return
  rosterMessage.value = ''
  try {
    const token = adminToken.value.trim() || undefined
    await deleteChild(token, childId)
    if (selectedChild.value === childId) {
      selectedChild.value = null
      await load()
    }
    rosterMessage.value = `已删除「${nickname}」，事件已匿名化`
    await loadChildren()
  } catch (e) {
    rosterMessage.value = `删除失败: ${e.message || e}（需管理口令）`
  }
}

// ---------------------------------------------------------------------------
// Recent detection batches: see pseudonyms & claim tracks to roster children
// ---------------------------------------------------------------------------

async function loadBatches() {
  batchesLoading.value = true
  claimMsg.value = ''
  try {
    const res = await fetchEventBatches(15)
    batches.value = res.batches || []
  } catch (e) {
    claimMsg.value = `检测记录加载失败: ${e.message || e}`
  } finally {
    batchesLoading.value = false
  }
}

function shortId(anonId) {
  if (!anonId) return '--'
  return anonId.length > 12 ? `${anonId.slice(0, 12)}…` : anonId
}

function batchTime(iso) {
  if (!iso) return '--'
  return new Date(iso).toLocaleString('zh-CN', {
    month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', hour12: false,
  })
}

function batchKindLabel(source) {
  return { image: '图片', video: '视频', demo: '演示', stream: '实时' }[source] || source
}

function claimKey(batch, track) {
  return `${batch.request_id}::${track.track_label}`
}

function trackTitle(track) {
  if (track.child_id) return `${track.child_nickname || '已认领'}`
  return `匿名 ${shortId(track.child_anon_id)}`
}

async function doClaim(batch, track) {
  claimMsg.value = ''
  const key = claimKey(batch, track)
  const sel = claimSel.value[key]
  let childId = sel
  const token = adminToken.value.trim() || undefined

  try {
    if (sel === '__new__') {
      const name = claimNewName.value.trim()
      if (!name) {
        claimMsg.value = '请填写新孩子昵称'
        return
      }
      const child = await createChild(token, { nickname: name, class_name: '' })
      childId = child.id
      await loadChildren()
    }
    if (!childId) {
      claimMsg.value = '请选择要认领的孩子'
      return
    }
    // Preferred: claim the cross-video anonymous identity (stable pseudonym).
    // Falls back to the request+track claim for legacy rows.
    if (track.child_anon_id) {
      const res = await claimIdentity(token, track.child_anon_id, childId)
      claimMsg.value = `已认领匿名身份 ${track.child_anon_id}（${res.linked_events} 条事件）→ 该孩子。以后这个代号再出现会自动归入。`
    } else {
      const res = await claimTrack(token, {
        child_id: childId,
        request_id: batch.request_id,
        track_label: track.track_label,
      })
      claimMsg.value = `已认领 ${res.linked_events} 条事件 → 该孩子（可去「按孩子查看」看个人日报）`
    }
    claimSel.value[key] = ''
    claimNewName.value = ''
    // Claim-to-burn: incinerate this track's session thumbnail immediately.
    if (track.has_thumb) {
      try {
        await burnTrackThumb(token, batch.request_id, track.track_label)
      } catch (e) {
        // burning is best-effort; the TTL will catch it anyway
      }
    }
    await Promise.all([loadBatches(), loadChildren(), loadIdentities(), load()])
  } catch (e) {
    claimMsg.value = `认领失败: ${e.message || e}（需管理口令）`
  }
}

async function burnBatchThumbs(batch) {
  claimMsg.value = ''
  const token = adminToken.value.trim() || undefined
  try {
    const res = await burnRequestThumbs(token, batch.request_id)
    claimMsg.value = `已焚毁本批次 ${res.removed} 张会话级缩略图（数据库不受影响）`
    await loadBatches()
  } catch (e) {
    claimMsg.value = `焚毁失败: ${e.message || e}（需管理口令）`
  }
}

async function doUnclaim(batch, track) {
  claimMsg.value = ''
  const token = adminToken.value.trim() || undefined
  try {
    if (track.child_anon_id) {
      const res = await unclaimIdentity(token, track.child_anon_id)
      claimMsg.value = `已解除，${res.unlinked_events} 条事件回到匿名`
    } else {
      const res = await unclaimTrack(token, track.child_id, {
        request_id: batch.request_id,
        track_label: track.track_label,
      })
      claimMsg.value = `已解除，${res.unlinked_events} 条事件回到匿名`
    }
    await Promise.all([loadBatches(), loadChildren(), loadIdentities(), load()])
  } catch (e) {
    claimMsg.value = `解除失败: ${e.message || e}（需管理口令）`
  }
}

// ---------------------------------------------------------------------------
// Anonymous identities & pending confirmations (re-ID)
// ---------------------------------------------------------------------------

async function loadIdentities() {
  identitiesLoading.value = true
  identityMsg.value = ''
  try {
    const [ids, pend] = await Promise.all([fetchIdentities(60), fetchPendingConfirmations(60)])
    identities.value = ids.identities || []
    pendingList.value = pend.pending || []
  } catch (e) {
    identityMsg.value = `身份列表加载失败: ${e.message || e}`
  } finally {
    identitiesLoading.value = false
  }
}

async function doIdentityClaim(identity) {
  identityMsg.value = ''
  const token = adminToken.value.trim() || undefined
  const sel = identityClaimSel.value[identity.anon_code]
  let childId = sel
  try {
    if (sel === '__new__') {
      const name = claimNewName.value.trim()
      if (!name) {
        identityMsg.value = '请填写新孩子昵称'
        return
      }
      const child = await createChild(token, { nickname: name, class_name: '' })
      childId = child.id
      await loadChildren()
    }
    if (!childId) {
      identityMsg.value = '请选择要认领的孩子'
      return
    }
    const res = await claimIdentity(token, identity.anon_code, childId)
    identityMsg.value = `已认领 ${identity.anon_code}（${res.linked_events} 条事件）→ 该孩子`
    identityClaimSel.value[identity.anon_code] = ''
    claimNewName.value = ''
    await Promise.all([loadIdentities(), loadChildren(), loadBatches(), load()])
  } catch (e) {
    identityMsg.value = `认领失败: ${e.message || e}（需管理口令）`
  }
}

async function doIdentityUnclaim(identity) {
  identityMsg.value = ''
  const token = adminToken.value.trim() || undefined
  try {
    const res = await unclaimIdentity(token, identity.anon_code)
    identityMsg.value = `已解除，${res.unlinked_events} 条事件回到匿名`
    await Promise.all([loadIdentities(), loadChildren(), loadBatches(), load()])
  } catch (e) {
    identityMsg.value = `解除失败: ${e.message || e}（需管理口令）`
  }
}

async function doIdentityDelete(identity) {
  identityMsg.value = ''
  const token = adminToken.value.trim() || undefined
  try {
    await deleteIdentity(token, identity.anon_code)
    identityMsg.value = `已删除特征原型 ${identity.anon_code}，其事件仍保留为匿名`
    await Promise.all([loadIdentities(), loadBatches()])
  } catch (e) {
    identityMsg.value = `删除失败: ${e.message || e}（需管理口令）`
  }
}

async function doResolvePending(pending, action) {
  identityMsg.value = ''
  claimMsg.value = ''
  const token = adminToken.value.trim() || undefined
  try {
    const pid = pending.id || pending.pending_id
    if (!pid) throw new Error('待确认记录缺少 ID')
    const res = await resolvePendingConfirmation(token, pid, action)
    const actionLabel = { link: '已合并到候选身份', keep: '已保留为独立匿名身份', dismiss: '已忽略' }[action]
    const msg = `${actionLabel}（${res.linked_events} 条事件处理）`
    // Show the result in BOTH panels: the user may be on either one.
    identityMsg.value = msg
    claimMsg.value = msg
    await Promise.all([loadIdentities(), loadBatches()])
  } catch (e) {
    const errText = e.message || e || ''
    const hint = String(errText).includes('401')
      ? '未授权（401）——请在本页上方填写管理口令后再操作'
      : `处理失败: ${errText}（需管理口令）`
    identityMsg.value = hint
    claimMsg.value = hint
  }
}

onMounted(async () => {
  await Promise.allSettled([fetchPrivacyStatus().then((p) => (privacy.value = p)), loadChildren()])
  await load()
})
</script>

<template>
  <section class="report-shell">
    <header class="report-topbar">
      <div>
        <p class="eyebrow">Privacy-First Reports</p>
        <h1>情绪日报 / 周报</h1>
      </div>
      <div class="report-actions">
        <span v-if="privacy" class="privacy-pill" :class="{ locked: privacy.access_control_enabled }">
          <Lock v-if="privacy.access_control_enabled" :size="14" />
          <Shield v-else :size="14" />
          {{
            privacy.stores_images
              ? '存图模式'
              : privacy.access_control_enabled
                ? '已启用访问控制 · 匿名化'
                : '匿名化 · 不存人脸图像'
          }}
        </span>
        <button class="btn ghost" @click="emit('back')">← 返回工作台</button>
      </div>
    </header>

    <div v-if="error" class="error-banner">⚠ {{ error }}</div>

    <!-- Period switcher -->
    <section class="period-bar">
      <div class="mode-tabs">
        <button :class="{ active: viewMode === 'daily' }" @click="switchMode('daily')">
          <CalendarDays :size="16" /> 日报
        </button>
        <button :class="{ active: viewMode === 'weekly' }" @click="switchMode('weekly')">
          <CalendarRange :size="16" /> 周报
        </button>
      </div>
      <div class="period-nav">
        <button class="btn small" @click="prevPeriod">← 上一期</button>
        <input
          v-if="viewMode === 'daily'"
          v-model="date"
          type="date"
          class="date-input"
          @change="load"
        />
        <input
          v-else
          v-model="weekStart"
          type="date"
          class="date-input"
          @change="load"
        />
        <button class="btn small" @click="nextPeriod">下一期 →</button>
        <button class="btn small" :disabled="loading" @click="load">
          <Loader2 v-if="loading" class="spin" :size="15" />
          <RefreshCw v-else :size="15" />
        </button>
      </div>
    </section>

    <!-- Child scope selector: aggregate or per-child daily view -->
    <section class="child-bar">
      <div class="child-scope">
        <span class="scope-label"><Users :size="15" /> 查看范围</span>
        <select v-model="selectedChild" class="child-select" @change="onChildChange">
          <option :value="null">全班 / 聚合统计</option>
          <option v-for="c in children" :key="c.id" :value="c.id">
            {{ c.nickname }}（{{ c.class_name || '未分班' }}）· {{ c.event_count }} 条
          </option>
        </select>
      </div>
      <div v-if="isPerChild" class="child-note">
        <Shield :size="14" />
        个人视图仅汇总教师已认领/关联的事件，未关联部分保持匿名
      </div>
      <button class="btn small" :class="{ active: rosterOpen }" @click="rosterOpen = !rosterOpen">
        <Users :size="15" /> 孩子档案
      </button>
      <button class="btn small" :class="{ active: batchesOpen }" @click="batchesOpen = !batchesOpen; if (batchesOpen && !batches.length) loadBatches()">
        <Database :size="15" /> 检测记录 / 认领
      </button>
      <button class="btn small" :class="{ active: identitiesOpen }" @click="identitiesOpen = !identitiesOpen; if (identitiesOpen && !identities.length) loadIdentities()">
        <Shield :size="15" /> 匿名身份
        <span v-if="pendingList.length" class="count-dot">{{ pendingList.length }}</span>
      </button>
    </section>

    <!-- Roster management (collapsible) -->
    <section v-if="rosterOpen" class="roster-panel">
      <div class="roster-form">
        <input v-model="newChildName" type="text" placeholder="昵称（必填），如：豆豆" class="token-input" />
        <input v-model="newChildClass" type="text" placeholder="班级（可选），如：小一班" class="token-input" />
        <button class="btn" @click="addChild">+ 登记孩子</button>
      </div>
      <p class="roster-tip">
        孩子档案 = 假名 ↔ 昵称映射（本地存储，不采集人脸）。删除档案后其关联事件自动匿名化。
        写操作需管理口令（下方数据管理区填写）。
      </p>
      <div v-if="children.length" class="roster-list">
        <div v-for="c in children" :key="c.id" class="roster-item">
          <div>
            <strong>{{ c.nickname }}</strong>
            <span>{{ c.class_name || '未分班' }} · {{ c.event_count }} 条事件</span>
          </div>
          <button class="btn small danger" @click="removeChild(c.id, c.nickname)">删除</button>
        </div>
      </div>
      <p v-else class="roster-empty">暂无档案。可在「数据管理」生成演示数据时自动创建示例档案，或手动登记。</p>
      <p v-if="rosterMessage" class="admin-msg">{{ rosterMessage }}</p>
    </section>

    <!-- Recent detection batches: pseudonym + claim -->
    <section v-if="batchesOpen" class="batches-panel">
      <div class="batches-head">
        <div>
          <p class="eyebrow">Recent Detections</p>
          <h2>检测记录 · 匿名代号与认领</h2>
        </div>
        <button class="btn small" :disabled="batchesLoading" @click="loadBatches">
          <Loader2 v-if="batchesLoading" class="spin" :size="14" />
          <RefreshCw v-else :size="14" />
        </button>
      </div>
      <p class="batches-tip">
        每次上传图片/视频会生成一个「批次」（request）。系统会用<b>匿名特征匹配</b>自动判断这个孩子之前是否出现过：
        高置信自动复用同一<b>匿名代号</b>（跨视频稳定），低置信自动新建代号，中间地带会出现少量「待确认」轨道。
        为了让你知道代号对应哪个孩子，每个轨道会显示一张<b>会话级缩略图</b>：它只存在于内存（不写入数据库、不落盘），
        <em>认领即焚</em>，2 小时后自动过期，重启进程即消失。老师看着脸把代号<em>认领</em>为档案里的孩子，
        之后该代号再次出现会自动归入其个人日报；解除或删除档案即回到匿名。
      </p>
      <div class="token-row" title="认领、待确认处理、焚毁等写操作需要管理口令">
        <KeyRound :size="13" />
        <input
          v-model="adminToken"
          type="password"
          placeholder="管理口令（认领 / 待确认 / 焚毁需要）"
          class="token-input compact"
          @keydown.enter="checkAuth"
        />
        <span v-if="authState === 'checking'" class="auth-state checking">校验中…</span>
        <span v-else-if="authState === 'ok'" class="auth-state ok">✓ 口令有效</span>
        <span v-else-if="authState === 'bad'" class="auth-state bad">✗ 口令无效</span>
        <span v-else class="auth-state idle">输入后自动校验</span>
      </div>
      <p v-if="claimMsg" class="admin-msg">{{ claimMsg }}</p>

      <div v-if="!batchesLoading && !batches.length" class="roster-empty">
        暂无检测记录。先在工作台上传图片/视频，再回来刷新查看。
      </div>

      <div v-for="batch in batches" :key="batch.request_id" class="batch-card">
        <div class="batch-meta">
          <strong>{{ batchKindLabel(batch.source) }}批次</strong>
          <span>{{ batchTime(batch.first_seen) }} · {{ batch.track_count }} 条轨道</span>
          <code class="req-id">{{ batch.request_id.slice(0, 8) }}</code>
          <button
            v-if="batch.tracks.some((t) => t.has_thumb)"
            class="btn tiny danger"
            title="焚毁本批次全部会话级缩略图（不写入数据库）"
            @click="burnBatchThumbs(batch)"
          >焚毁本批次缩略图</button>
        </div>
        <div v-for="track in batch.tracks" :key="track.track_label + track.child_anon_id" class="track-row">
          <!-- Session-only claim thumbnail (RAM buffer, burn-on-claim). The teacher
               recognises the child from this face; the DB never stores any image. -->
          <img
            v-if="track.has_thumb && !track.child_id"
            :src="thumbUrl(batch.request_id, track.track_label)"
            class="thumb-avatar"
            :alt="track.child_anon_id"
            loading="lazy"
          />
          <!-- Identity shown to operators: full pseudonym or claimed nickname. The
               internal person-N track label stays hidden (title tooltip only). -->
          <span v-if="track.child_id" class="anon-chip linked" :title="track.track_label">
            <CheckCircle2 :size="11" />
            {{ track.child_nickname }}（{{ track.child_class || '未分班' }}）
          </span>
          <span v-else class="anon-chip" :title="track.track_label">
            <Shield :size="11" />
            {{ track.child_anon_id }}
          </span>
          <span v-if="track.sample_count > 1" class="sample-note">{{ track.sample_count }} 次采样</span>
          <span class="track-emotion" :style="{ color: emotionColor(track.emotion) }">
            {{ LABELS[track.emotion] || track.emotion }} {{ Math.round(track.confidence * 100) }}%
          </span>
          <span v-if="track.alert_triggered" class="neg-badge">⚠ 告警</span>
          <span v-if="track.pending" class="pending-badge" :title="'与候选身份相似度 ' + Math.round(track.pending.similarity * 100) + '%'">
            ❓ 待确认 {{ Math.round(track.pending.similarity * 100) }}%
          </span>

          <template v-if="!track.child_id">
            <select v-model="claimSel[claimKey(batch, track)]" class="claim-select">
              <option value="">认领为…</option>
              <option v-for="c in children" :key="c.id" :value="c.id">{{ c.nickname }}（{{ c.class_name || '未分班' }}）</option>
              <option value="__new__">＋ 新建孩子…</option>
            </select>
            <input
              v-if="claimSel[claimKey(batch, track)] === '__new__'"
              v-model="claimNewName"
              type="text"
              placeholder="新孩子昵称"
              class="new-name-input"
            />
            <button
              class="btn small"
              :disabled="!claimSel[claimKey(batch, track)]"
              @click="doClaim(batch, track)"
            >认领</button>
          </template>
          <template v-else>
            <button class="btn small danger" @click="doUnclaim(batch, track)">解除</button>
          </template>

          <div v-if="track.pending && !track.child_id" class="pending-actions">
            <template v-if="track.pending.candidate_anon_code">
              <button class="btn tiny" @click="doResolvePending(track.pending, 'link')">
                合并到 {{ track.pending.candidate_nickname || track.pending.candidate_anon_code }}
              </button>
            </template>
            <button class="btn tiny" @click="doResolvePending(track.pending, 'keep')">保留为独立匿名身份</button>
            <button class="btn tiny" @click="doResolvePending(track.pending, 'dismiss')">忽略</button>
          </div>
        </div>
      </div>
    </section>

    <!-- Anonymous identities: cross-video re-ID gallery + pending confirmations -->
    <section v-if="identitiesOpen" class="batches-panel">
      <div class="batches-head">
        <div>
          <p class="eyebrow">Anonymous Identity Gallery</p>
          <h2>匿名身份 · 特征匹配图库</h2>
        </div>
        <button class="btn small" :disabled="identitiesLoading" @click="loadIdentities">
          <Loader2 v-if="identitiesLoading" class="spin" :size="14" />
          <RefreshCw v-else :size="14" />
        </button>
      </div>
      <p class="batches-tip">
        每个匿名身份保存一个<b>不可逆的 768 维特征原型</b>（不存人脸图像、视频或文件名）。
        孩子离开镜头后再次出现时，系统按特征相似度自动判断：高置信复用同一代号，灰色地带生成「待确认」。
        特征原型遵循与事件相同的留存期限，可一键删除（事件仍保留为匿名）。
      </p>
      <div class="token-row" title="认领、待确认处理、删除原型等写操作需要管理口令">
        <KeyRound :size="13" />
        <input
          v-model="adminToken"
          type="password"
          placeholder="管理口令（认领 / 待确认 / 删除需要）"
          class="token-input compact"
          @keydown.enter="checkAuth"
        />
        <span v-if="authState === 'checking'" class="auth-state checking">校验中…</span>
        <span v-else-if="authState === 'ok'" class="auth-state ok">✓ 口令有效</span>
        <span v-else-if="authState === 'bad'" class="auth-state bad">✗ 口令无效</span>
        <span v-else class="auth-state idle">输入后自动校验</span>
      </div>
      <p v-if="identityMsg" class="admin-msg">{{ identityMsg }}</p>

      <div v-if="pendingList.length" class="pending-block">
        <h3>待确认（{{ pendingList.length }}）· 轨道级</h3>
        <div v-for="p in pendingList" :key="p.id" class="pending-row">
          <span class="anon-chip"><Shield :size="11" /> {{ p.anon_code }}</span>
          <span class="pending-info">
            相似度 {{ Math.round((p.similarity || 0) * 100) }}% · 候选
            <b>{{ p.candidate_nickname || p.candidate_anon_code || '无' }}</b>
          </span>
          <span class="pending-actions">
            <button v-if="p.candidate_anon_code" class="btn tiny" @click="doResolvePending(p, 'link')">合并为候选</button>
            <button class="btn tiny" @click="doResolvePending(p, 'keep')">保留独立</button>
            <button class="btn tiny" @click="doResolvePending(p, 'dismiss')">忽略</button>
          </span>
        </div>
      </div>

      <div v-if="!identitiesLoading && !identities.length" class="roster-empty">
        暂无匿名身份。上传一段视频后，系统会为每个新出现的孩子建立匿名身份。
      </div>

      <div v-for="i in identities" :key="i.anon_code" class="identity-row">
        <span class="anon-chip" :class="{ linked: i.child_id }"><Shield :size="11" /> {{ i.anon_code }}</span>
        <span class="sample-note">
          {{ i.child_id ? `${i.child_nickname}（${i.child_class || '未分班'}）` : '未认领' }}
          · {{ i.event_count }} 条事件 · {{ i.sample_count }} 次采样
        </span>
        <template v-if="!i.child_id">
          <select v-model="identityClaimSel[i.anon_code]" class="claim-select">
            <option value="">认领为…</option>
            <option v-for="c in children" :key="c.id" :value="c.id">{{ c.nickname }}（{{ c.class_name || '未分班' }}）</option>
            <option value="__new__">＋ 新建孩子…</option>
          </select>
          <input
            v-if="identityClaimSel[i.anon_code] === '__new__'"
            v-model="claimNewName"
            type="text"
            placeholder="新孩子昵称"
            class="new-name-input"
          />
          <button class="btn small" :disabled="!identityClaimSel[i.anon_code]" @click="doIdentityClaim(i)">认领</button>
        </template>
        <template v-else>
          <button class="btn small danger" @click="doIdentityUnclaim(i)">解除</button>
        </template>
        <button class="btn tiny danger" @click="doIdentityDelete(i)">删原型</button>
      </div>
    </section>

    <template v-if="activeData">
      <!-- Headline summary -->
      <section class="summary-cards">
        <article class="s-card">
          <div class="s-label">总情绪事件</div>
          <strong>{{ totalEvents }}</strong>
          <span class="s-sub">
            较上一{{ viewMode === 'daily' ? '天' : '周' }}
            <em :class="deltaText(activeData.vs_previous_day || activeData.vs_previous_week, 'total_events_delta')?.cls">
              {{ deltaText(activeData.vs_previous_day || activeData.vs_previous_week, 'total_events_delta')?.text || '--' }}
            </em>
          </span>
        </article>
        <article class="s-card accent">
          <div class="s-label">主导情绪</div>
          <strong :style="{ color: mainEmotionColor }">{{ mainEmotion }}</strong>
          <span class="s-sub">按事件计数</span>
        </article>
        <article class="s-card warn">
          <div class="s-label">负面情绪事件</div>
          <strong>{{ negativeEvents }}</strong>
          <span class="s-sub">
            占比 {{ negativeRatio }}%
            <em :class="deltaText(activeData.vs_previous_day || activeData.vs_previous_week, 'negative_events_delta')?.cls">
              {{ deltaText(activeData.vs_previous_day || activeData.vs_previous_week, 'negative_events_delta')?.text || '' }}
            </em>
          </span>
        </article>
        <article class="s-card">
          <div class="s-label">{{ isPerChild ? '档案孩子' : '涉及幼儿（匿名）' }}</div>
          <strong v-if="isPerChild">{{ viewingChild?.nickname || '--' }}</strong>
          <strong v-else>{{ distinctChildren }}</strong>
          <span class="s-sub">
            <template v-if="isPerChild">{{ viewingChild?.class_name || '未分班' }} · {{ alertCount }} 次告警</template>
            <template v-else>告警 {{ alertCount }} 次</template>
          </span>
        </article>
      </section>

      <!-- Charts -->
      <section class="chart-grid">
        <article class="panel">
          <div class="panel-head compact">
            <div>
              <p class="eyebrow">Emotion Distribution</p>
              <h2>情绪分布</h2>
            </div>
          </div>
          <div class="bar-list">
            <div v-for="row in emotionRows" :key="row.key" class="bar-row">
              <span class="bar-label" :title="row.label">{{ row.label }}</span>
              <div class="bar-track">
                <div
                  class="bar-fill"
                  :style="{ width: Math.max(row.count ? 4 : 0, row.pct) + '%', background: row.color }"
                ></div>
              </div>
              <b>{{ row.count }}</b>
            </div>
          </div>
        </article>

        <article class="panel">
          <div class="panel-head compact">
            <div>
              <p class="eyebrow">Hourly Activity</p>
              <h2>时段分布（UTC 小时）</h2>
            </div>
          </div>
          <div class="hour-chart">
            <div v-for="h in hourlyRows" :key="h.hour" class="hour-col">
              <div class="hour-bar-wrap">
                <div class="hour-bar" :style="{ height: Math.max(h.count ? 6 : 0, h.pct) + '%' }"></div>
              </div>
              <span class="hour-label">{{ h.hour }}</span>
            </div>
          </div>
          <p class="panel-note">峰值时段: {{ activeData.summary?.peak_hour ?? '--' }} 时（UTC）</p>
        </article>
      </section>

      <!-- Weekly series (weekly mode only) -->
      <section v-if="viewMode === 'weekly' && weekly" class="panel series-panel">
        <div class="panel-head compact">
          <div>
            <p class="eyebrow">Daily Series</p>
            <h2>每日事件数</h2>
          </div>
        </div>
        <div class="series-row">
          <div v-for="d in dailySeries" :key="d.date" class="series-col">
            <div class="series-bar-wrap">
              <div
                class="series-bar"
                :class="{ negative: d.negative_events > 0 }"
                :style="{ height: d.pct + '%' }"
                :title="`${d.date} 事件 ${d.total_events} / 负面 ${d.negative_events}`"
              ></div>
            </div>
            <span class="series-label">周{{ d.weekday }}</span>
            <b>{{ d.total_events }}</b>
          </div>
        </div>
      </section>

      <!-- Export -->
      <section class="toolbar">
        <button class="btn" @click="exportReport">
          <Download :size="16" /> 导出 JSON
        </button>
        <span v-if="activeData.tz_note" class="muted-note">{{ activeData.tz_note }}</span>
      </section>
    </template>

    <div v-else-if="!loading && !error" class="empty-state">
      <BarChart3 :size="40" />
      <p>该时段暂无数据。点击下方「生成演示数据」可填充近 7 天示例。</p>
    </div>

    <!-- Admin / demo tools -->
    <section class="admin-panel">
      <div class="admin-head" @click="adminOpen = !adminOpen">
        <Database :size="16" />
        <span>数据管理（演示数据 / 清除）</span>
        <Sparkles :size="15" class="admin-toggle" :class="{ open: adminOpen }" />
      </div>
      <div v-if="adminOpen" class="admin-body">
        <p class="admin-tip">
          生成的数据带有 <code>source=demo</code> 标记，仅用于填充报表演示；清除操作会删除全部历史事件。
          若后端启用了访问控制（EMOTION_API_TOKEN），管理操作需填写口令。
        </p>
        <div class="admin-row">
          <input v-model="adminToken" type="password" placeholder="管理口令（可选）" class="token-input" />
          <button class="btn" @click="adminAction('seed')">
            <FlaskConical :size="15" /> 生成近7天演示数据
          </button>
          <button class="btn danger" @click="adminAction('purge')">
            <Trash2 :size="15" /> 清除全部事件
          </button>
        </div>
        <p v-if="adminMessage" class="admin-msg">{{ adminMessage }}</p>
        <p v-if="privacy" class="privacy-detail">
          <KeyRound :size="13" />
          存储策略：仅匿名情绪事件（{{ privacy.storage?.events ?? 0 }} 条），
          保留 {{ privacy.retention_days }} 天自动清理；
          不存图片 / 不存人脸框；假名算法 {{ privacy.pseudonym_algorithm }}
        </p>
      </div>
    </section>
  </section>
</template>

<style scoped>
.report-shell {
  min-height: 100vh;
  box-sizing: border-box;
  padding: 22px 26px 40px;
  display: flex;
  flex-direction: column;
  gap: 18px;
  background:
    radial-gradient(1100px 500px at 85% -10%, rgba(93, 132, 245, 0.14), transparent 60%),
    radial-gradient(900px 420px at -10% 10%, rgba(24, 166, 148, 0.10), transparent 55%),
    #0b1117;
  color: #dfe7ef;
}
.report-topbar {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 16px;
  flex-wrap: wrap;
}
.report-topbar h1 { margin: 0; font-size: 24px; color: #f3f6fa; }
.eyebrow { margin: 0 0 4px; font-size: 11px; letter-spacing: 0.18em; text-transform: uppercase; color: #7e92a6; }
.report-actions { display: flex; align-items: center; gap: 12px; }
.privacy-pill {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 7px 12px; border-radius: 999px; font-size: 12px;
  background: rgba(94, 135, 150, 0.16); border: 1px solid rgba(94, 135, 150, 0.4); color: #9fc3d2;
}
.privacy-pill.locked { background: rgba(221, 95, 85, 0.14); border-color: rgba(221, 95, 85, 0.45); color: #eaa09a; }

.period-bar { display: flex; justify-content: space-between; align-items: center; gap: 14px; flex-wrap: wrap; }
.mode-tabs { display: flex; gap: 6px; background: rgba(255,255,255,0.05); padding: 4px; border-radius: 10px; }
.mode-tabs button {
  display: inline-flex; align-items: center; gap: 6px; border: none; cursor: pointer;
  padding: 8px 16px; border-radius: 8px; background: transparent; color: #93a6b8; font-size: 13px;
}
.mode-tabs button.active { background: #1d2a3a; color: #fff; box-shadow: inset 0 0 0 1px rgba(255,255,255,0.08); }
.period-nav { display: flex; align-items: center; gap: 8px; }

/* Child scope + roster */
.child-bar {
  display: flex; align-items: center; gap: 12px; flex-wrap: wrap;
  background: rgba(255,255,255,0.035); border: 1px solid rgba(255,255,255,0.08);
  border-radius: 12px; padding: 10px 14px;
}
.child-scope { display: flex; align-items: center; gap: 10px; }
.scope-label { display: inline-flex; align-items: center; gap: 6px; font-size: 13px; color: #b7c6d4; }
.child-select {
  background: #141e29; border: 1px solid #2a3a4c; color: #dfe7ef;
  padding: 7px 10px; border-radius: 8px; font-size: 13px; min-width: 230px;
}
.child-note { display: inline-flex; align-items: center; gap: 6px; font-size: 12px; color: #8fb8a8; }
.btn.active { outline: 1px solid rgba(232, 184, 75, 0.55); }

.roster-panel {
  background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08);
  border-radius: 12px; padding: 14px 16px;
}
.roster-form { display: flex; gap: 10px; flex-wrap: wrap; }
.roster-tip { font-size: 12px; color: #7d91a4; margin: 10px 0 6px; line-height: 1.6; }
.roster-list { display: flex; flex-direction: column; gap: 6px; margin-top: 6px; }
.roster-item {
  display: flex; justify-content: space-between; align-items: center;
  background: rgba(255,255,255,0.045); border-radius: 10px; padding: 9px 12px;
}
.roster-item strong { color: #eef2f6; font-size: 14px; }
.roster-item span { color: #7d91a4; font-size: 12px; margin-left: 8px; }
.roster-empty { font-size: 12px; color: #6f8396; margin-top: 6px; }

/* Detection batches + claim */
.batches-panel {
  background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08);
  border-radius: 12px; padding: 14px 16px;
}
.batches-head { display: flex; justify-content: space-between; align-items: center; }
.batches-head h2 { margin: 0; font-size: 15px; color: #eef2f6; }
.batches-tip { font-size: 12px; color: #8fa2b5; line-height: 1.7; margin: 8px 0 4px; }
.batches-tip b { color: #cfe3f0; font-family: ui-monospace, monospace; }
.batches-tip em { color: #e8b84b; font-style: normal; }
.batch-card {
  background: rgba(255,255,255,0.035); border-radius: 10px;
  margin-top: 10px; padding: 10px 12px;
}
.batch-meta {
  display: flex; align-items: center; gap: 12px;
  font-size: 12px; color: #7d91a4; padding-bottom: 6px;
  border-bottom: 1px dashed rgba(255,255,255,0.08); flex-wrap: wrap;
}
.batch-meta strong { color: #c7d6e3; font-size: 13px; }
.req-id { font-size: 11px; color: #6f8396; background: rgba(255,255,255,0.06); padding: 1px 6px; border-radius: 5px; }
.track-row {
  display: flex; align-items: center; gap: 10px; flex-wrap: wrap;
  padding: 8px 2px; border-bottom: 1px solid rgba(255,255,255,0.04);
}
.track-row:last-child { border-bottom: none; }
.track-id {
  display: none; /* internal person-N label is hidden from operators */
}
.track-id.claimed { display: none; }
.anon-chip {
  display: inline-flex; align-items: center; gap: 5px;
  font-family: ui-monospace, monospace; font-size: 12px;
  color: #9fc3d2; background: rgba(94,135,150,0.12); padding: 2px 9px; border-radius: 999px;
  max-width: 190px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.anon-chip.linked { color: #7fd8b9; background: rgba(99,199,168,0.14); }
.sample-note { font-size: 11px; color: #5f7386; }
.thumb-avatar {
  width: 40px; height: 40px; border-radius: 50%; object-fit: cover;
  border: 2px solid rgba(94,135,150,0.55); flex: none;
  background: rgba(255,255,255,0.05);
}
.pending-badge { font-size: 11px; color: #e9c46a; border: 1px solid rgba(233,196,106,0.55); padding: 0 7px; border-radius: 999px; white-space: nowrap; }
.pending-actions { display: flex; gap: 6px; align-items: center; margin-left: auto; flex-wrap: wrap; }
.pending-row, .identity-row {
  display: flex; align-items: center; gap: 10px; flex-wrap: wrap;
  background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06);
  border-radius: 8px; padding: 8px 10px; margin-bottom: 8px;
}
.pending-block { margin-bottom: 14px; }
.pending-block h3 { margin: 0 0 8px; font-size: 13px; color: #e9c46a; }
.pending-info { font-size: 12px; color: #9fb2c3; }
.count-dot {
  background: #e9c46a; color: #1b2735; border-radius: 999px; font-size: 10px;
  padding: 0 6px; line-height: 16px; font-weight: 700;
}
.btn.tiny { padding: 2px 8px; font-size: 11px; }
.track-emotion { font-size: 12px; min-width: 80px; }
.neg-badge { font-size: 10px; color: #eaa09a; border: 1px solid rgba(221,95,85,0.5); padding: 0 6px; border-radius: 999px; }
.claim-select {
  background: #141e29; border: 1px solid #2a3a4c; color: #dfe7ef;
  padding: 5px 8px; border-radius: 8px; font-size: 12px;
}
.new-name-input {
  background: #141e29; border: 1px solid #2a3a4c; color: #dfe7ef;
  padding: 5px 8px; border-radius: 8px; font-size: 12px; width: 110px;
}

.btn.danger {
  background: rgba(221, 95, 85, 0.15); border-color: rgba(221, 95, 85, 0.5); color: #eaa09a;
}
.btn.danger:hover { background: rgba(221, 95, 85, 0.25); }
.date-input {
  background: #141e29; border: 1px solid #2a3a4c; color: #dfe7ef;
  padding: 7px 10px; border-radius: 8px; font-size: 13px;
}
.btn {
  display: inline-flex; align-items: center; gap: 7px; cursor: pointer;
  background: #1c2a3a; color: #dfe7ef; border: 1px solid #31455c;
  padding: 8px 14px; border-radius: 9px; font-size: 13px;
}
.btn:hover { background: #24364a; }
.btn.small { padding: 6px 10px; font-size: 12px; }
.btn.ghost { background: transparent; }
.btn.danger { background: rgba(221, 95, 85, 0.15); border-color: rgba(221, 95, 85, 0.5); color: #eaa09a; }
.btn:disabled { opacity: 0.5; cursor: default; }
.spin { animation: rot 1s linear infinite; }
@keyframes rot { to { transform: rotate(360deg); } }

.error-banner {
  background: rgba(221, 95, 85, 0.14); border: 1px solid rgba(221, 95, 85, 0.45);
  color: #f0b0aa; padding: 10px 14px; border-radius: 10px; font-size: 13px;
}

.summary-cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; }
.s-card {
  background: linear-gradient(160deg, rgba(255,255,255,0.055), rgba(255,255,255,0.02));
  border: 1px solid rgba(255,255,255,0.08); border-radius: 14px; padding: 16px 18px;
  display: flex; flex-direction: column; gap: 6px;
}
.s-card.accent { border-top: 3px solid #e8b84b; }
.s-card.warn { border-top: 3px solid #dd5f55; }
.s-label { font-size: 12px; color: #8fa2b5; }
.s-card strong { font-size: 30px; line-height: 1; color: #f3f6fa; }
.s-sub { font-size: 12px; color: #7d91a4; display: flex; align-items: center; gap: 4px; }
.s-sub em { font-style: normal; font-weight: 700; }
.s-sub em.up { color: #e07167; }
.s-sub em.down { color: #63c7a8; }
.s-sub em.flat { color: #8fa2b5; }

.chart-grid { display: grid; grid-template-columns: 1.2fr 1fr; gap: 14px; }
.panel {
  background: linear-gradient(160deg, rgba(255,255,255,0.045), rgba(255,255,255,0.015));
  border: 1px solid rgba(255,255,255,0.08); border-radius: 14px; padding: 16px 18px;
}
.panel-head.compact { margin-bottom: 12px; }
.panel-head h2 { margin: 0; font-size: 15px; color: #eef2f6; }
.bar-list { display: flex; flex-direction: column; gap: 9px; }
.bar-row { display: grid; grid-template-columns: 44px 1fr 34px; align-items: center; gap: 10px; }
.bar-label { font-size: 13px; color: #b7c6d4; }
.bar-track { height: 16px; background: rgba(255,255,255,0.06); border-radius: 6px; overflow: hidden; }
.bar-fill { height: 100%; border-radius: 6px; transition: width 0.4s ease; }
.bar-row b { text-align: right; font-size: 13px; color: #dfe7ef; }

.hour-chart { display: flex; align-items: flex-end; gap: 3px; height: 120px; }
.hour-col { flex: 1; display: flex; flex-direction: column; align-items: center; gap: 4px; height: 100%; }
.hour-bar-wrap { flex: 1; width: 100%; display: flex; align-items: flex-end; }
.hour-bar {
  width: 100%; min-height: 2px; border-radius: 3px 3px 0 0;
  background: linear-gradient(180deg, #5b8dd6, #35609f);
}
.hour-label { font-size: 9px; color: #6f8396; }
.panel-note { margin: 10px 0 0; font-size: 12px; color: #7d91a4; }

.series-panel { margin-top: 2px; }
.series-row { display: flex; gap: 10px; height: 150px; align-items: flex-end; }
.series-col { flex: 1; display: flex; flex-direction: column; align-items: center; gap: 5px; height: 100%; }
.series-bar-wrap { flex: 1; width: 70%; display: flex; align-items: flex-end; }
.series-bar { width: 100%; border-radius: 6px 6px 0 0; background: linear-gradient(180deg, #e8b84b, #a97e22); }
.series-bar.negative { background: linear-gradient(180deg, #dd5f55, #96332c); }
.series-label { font-size: 12px; color: #93a6b8; }
.series-col b { font-size: 12px; color: #dfe7ef; }

.toolbar { display: flex; align-items: center; gap: 12px; }
.muted-note { font-size: 11px; color: #5f7386; }
.empty-state {
  border: 1px dashed rgba(255,255,255,0.16); border-radius: 14px; padding: 40px;
  display: flex; flex-direction: column; align-items: center; gap: 10px; color: #7d91a4; text-align: center;
}

.admin-panel {
  border: 1px solid rgba(255,255,255,0.09); border-radius: 14px; overflow: hidden;
  background: rgba(255,255,255,0.025);
}
.admin-head {
  display: flex; align-items: center; gap: 8px; padding: 13px 16px; cursor: pointer;
  font-size: 13px; color: #b7c6d4; user-select: none;
}
.admin-toggle { margin-left: auto; transition: transform 0.2s; }
.admin-toggle.open { transform: rotate(45deg); }
.admin-body { padding: 4px 16px 16px; border-top: 1px solid rgba(255,255,255,0.06); }
.admin-tip { font-size: 12px; color: #7d91a4; line-height: 1.6; }
.admin-tip code { background: rgba(255,255,255,0.08); padding: 1px 6px; border-radius: 5px; }
.admin-row { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 12px; }
.token-input {
  background: #141e29; border: 1px solid #2a3a4c; color: #dfe7ef;
  padding: 8px 12px; border-radius: 9px; font-size: 13px; min-width: 200px;
}
.token-input.compact { min-width: 0; width: 280px; max-width: 100%; padding: 6px 10px; font-size: 12px; }
.token-row { display: flex; align-items: center; gap: 8px; margin: 10px 0 2px; color: #9fb2c3; font-size: 12px; }
.auth-state { font-size: 12px; white-space: nowrap; }
.auth-state.ok { color: #7fd8b9; }
.auth-state.bad { color: #eaa09a; }
.auth-state.checking { color: #e9c46a; }
.auth-state.idle { color: #5f7386; }
.admin-msg { margin-top: 10px; font-size: 12px; color: #9fc3d2; }
.privacy-detail {
  margin-top: 12px; font-size: 11px; color: #6f8396;
  display: flex; align-items: flex-start; gap: 6px; line-height: 1.6;
}

@media (max-width: 1100px) {
  .chart-grid { grid-template-columns: 1fr; }
}
</style>
