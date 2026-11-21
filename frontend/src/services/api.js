const API_BASE = import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8002'

export const emotionLabels = {
  anger: '生气',
  contempt: '轻蔑',
  disgust: '厌恶',
  fear: '害怕',
  happy: '开心',
  neutral: '平静',
  sad: '伤心',
  surprise: '惊讶',
}

export const negativeEmotions = new Set(['anger', 'disgust', 'fear', 'sad'])

export async function fetchHealth() {
  const res = await fetch(`${API_BASE}/health`)
  if (!res.ok) throw new Error(`health ${res.status}`)
  return res.json()
}

export async function checkAdminToken(token) {
  // Probe the token-protected endpoint to show live password validity.
  const res = await fetch(`${API_BASE}/health/admin`, {
    headers: token ? { Authorization: `Bearer ${token}` } : {},
  })
  if (res.ok) {
    const data = await res.json()
    return { authorized: true, access_control_enabled: data.access_control_enabled }
  }
  if (res.status === 401) return { authorized: false, access_control_enabled: true }
  throw new Error(`check ${res.status}`)
}

export async function fetchCameras() {
  const res = await fetch(`${API_BASE}/api/cameras`)
  if (!res.ok) throw new Error(`cameras ${res.status}`)
  return res.json()
}

export async function fetchAlerts() {
  const res = await fetch(`${API_BASE}/api/alerts`)
  if (!res.ok) throw new Error(`alerts ${res.status}`)
  return res.json()
}

export async function fetchStats() {
  const res = await fetch(`${API_BASE}/api/alerts/stats/summary`)
  if (!res.ok) throw new Error(`stats ${res.status}`)
  return res.json()
}

export async function fetchDailyReport(date, { compare = true } = {}) {
  const q = new URLSearchParams()
  if (date) q.set('date', date)
  if (compare === false) q.set('compare', 'false')
  const res = await fetch(`${API_BASE}/api/reports/daily?${q.toString()}`)
  if (!res.ok) throw new Error(`daily report ${res.status}`)
  return res.json()
}

export async function fetchWeeklyReport(start, { compare = true } = {}) {
  const q = new URLSearchParams()
  if (start) q.set('start', start)
  if (compare === false) q.set('compare', 'false')
  const res = await fetch(`${API_BASE}/api/reports/weekly?${q.toString()}`)
  if (!res.ok) throw new Error(`weekly report ${res.status}`)
  return res.json()
}

export async function fetchPrivacyStatus() {
  const res = await fetch(`${API_BASE}/api/reports/privacy`)
  if (!res.ok) throw new Error(`privacy ${res.status}`)
  return res.json()
}

export async function purgeAllEvents(token) {
  const res = await fetch(`${API_BASE}/api/reports/purge`, {
    method: 'POST',
    headers: token ? { Authorization: `Bearer ${token}` } : {},
  })
  if (!res.ok) throw new Error(`purge ${res.status}`)
  return res.json()
}

export async function fetchChildren(includeDisabled = true) {
  const res = await fetch(`${API_BASE}/api/children${includeDisabled ? '' : '?include_disabled=false'}`)
  if (!res.ok) throw new Error(`children ${res.status}`)
  return res.json()
}

export async function createChild(token, { nickname, class_name = '', note = '' }) {
  const res = await fetch(`${API_BASE}/api/children`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    body: JSON.stringify({ nickname, class_name, note }),
  })
  if (!res.ok) throw new Error(`create child ${res.status}`)
  return res.json()
}

export async function deleteChild(token, childId) {
  const res = await fetch(`${API_BASE}/api/children/${childId}`, {
    method: 'DELETE',
    headers: token ? { Authorization: `Bearer ${token}` } : {},
  })
  if (!res.ok) throw new Error(`delete child ${res.status}`)
  return res.json()
}

export async function fetchChildDaily(childId, date) {
  const q = new URLSearchParams()
  if (date) q.set('date', date)
  const res = await fetch(`${API_BASE}/api/children/${childId}/daily?${q.toString()}`)
  if (!res.ok) throw new Error(`child daily ${res.status}`)
  return res.json()
}

export async function fetchEventBatches(limit = 20) {
  const res = await fetch(`${API_BASE}/api/events?limit=${limit}`)
  if (!res.ok) throw new Error(`events ${res.status}`)
  return res.json()
}

export async function claimTrack(token, { child_id, request_id, track_label, anon_code }) {
  const res = await fetch(`${API_BASE}/api/children/claim`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    body: JSON.stringify({ child_id, request_id, track_label, anon_code }),
  })
  if (!res.ok) throw new Error(`claim ${res.status}`)
  return res.json()
}

export async function unclaimTrack(token, childId, { request_id, track_label }) {
  const res = await fetch(`${API_BASE}/api/children/${childId}/unclaim`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    body: JSON.stringify({ request_id, track_label }),
  })
  if (!res.ok) throw new Error(`unclaim ${res.status}`)
  return res.json()
}

// --------------------------------------------------------------------------
// Anonymous identities (cross-video re-ID) and pending confirmations
// --------------------------------------------------------------------------

export async function fetchIdentities(limit = 100) {
  const res = await fetch(`${API_BASE}/api/identities?limit=${limit}`)
  if (!res.ok) throw new Error(`identities ${res.status}`)
  return res.json()
}

export async function fetchPendingConfirmations(limit = 100) {
  const res = await fetch(`${API_BASE}/api/identities/pending?limit=${limit}`)
  if (!res.ok) throw new Error(`pending ${res.status}`)
  return res.json()
}

export async function resolvePendingConfirmation(token, pendingId, action) {
  const res = await fetch(`${API_BASE}/api/identities/pending/${pendingId}/resolve`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    body: JSON.stringify({ action }),
  })
  if (!res.ok) throw new Error(`resolve ${res.status}`)
  return res.json()
}

export async function claimIdentity(token, anonCode, childId) {
  const res = await fetch(`${API_BASE}/api/identities/${encodeURIComponent(anonCode)}/claim`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    body: JSON.stringify({ child_id: childId }),
  })
  if (!res.ok) throw new Error(`claim identity ${res.status}`)
  return res.json()
}

export async function unclaimIdentity(token, anonCode) {
  const res = await fetch(`${API_BASE}/api/identities/${encodeURIComponent(anonCode)}/unclaim`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    body: JSON.stringify({}),
  })
  if (!res.ok) throw new Error(`unclaim identity ${res.status}`)
  return res.json()
}

export async function deleteIdentity(token, anonCode) {
  const res = await fetch(`${API_BASE}/api/identities/${encodeURIComponent(anonCode)}`, {
    method: 'DELETE',
    headers: token ? { Authorization: `Bearer ${token}` } : {},
  })
  if (!res.ok) throw new Error(`delete identity ${res.status}`)
  return res.json()
}

// --------------------------------------------------------------------------
// Session-only claim thumbnails (RAM buffer; burn on claim)
// --------------------------------------------------------------------------

export function thumbUrl(requestId, trackLabel) {
  return `${API_BASE}/api/thumbs/${encodeURIComponent(requestId)}/${encodeURIComponent(trackLabel)}`
}

export async function burnTrackThumb(token, requestId, trackLabel) {
  const res = await fetch(`${API_BASE}/api/thumbs/${encodeURIComponent(requestId)}/${encodeURIComponent(trackLabel)}`, {
    method: 'DELETE',
    headers: token ? { Authorization: `Bearer ${token}` } : {},
  })
  if (!res.ok) throw new Error(`burn thumb ${res.status}`)
  return res.json()
}

export async function burnRequestThumbs(token, requestId) {
  const res = await fetch(`${API_BASE}/api/thumbs/${encodeURIComponent(requestId)}`, {
    method: 'DELETE',
    headers: token ? { Authorization: `Bearer ${token}` } : {},
  })
  if (!res.ok) throw new Error(`burn thumbs ${res.status}`)
  return res.json()
}

export async function seedDemoEvents(token, { days = 7, perDay = 40 } = {}) {
  const res = await fetch(`${API_BASE}/api/reports/seed?days=${days}&per_day=${perDay}`, {
    method: 'POST',
    headers: token ? { Authorization: `Bearer ${token}` } : {},
  })
  if (!res.ok) throw new Error(`seed ${res.status}`)
  return res.json()
}

export async function inferImage(file) {
  const form = new FormData()
  form.append('file', file)
  const res = await fetch(`${API_BASE}/api/infer/image`, { method: 'POST', body: form })
  if (!res.ok) {
    const text = await res.text()
    throw new Error(text || `infer ${res.status}`)
  }
  return res.json()
}

export async function inferVideo(file) {
  const form = new FormData()
  form.append('file', file)
  const res = await fetch(`${API_BASE}/api/infer/video`, { method: 'POST', body: form })
  if (!res.ok) {
    const text = await res.text()
    throw new Error(text || `video ${res.status}`)
  }
  return res.json()
}

export function openStream(onMessage, onStatus) {
  const url = API_BASE.replace(/^http/, 'ws') + '/ws/stream'
  const ws = new WebSocket(url)
  ws.onopen = () => onStatus?.('connected')
  ws.onclose = () => onStatus?.('closed')
  ws.onerror = () => onStatus?.('error')
  ws.onmessage = (event) => {
    try {
      onMessage?.(JSON.parse(event.data))
    } catch {
      onMessage?.({ type: 'raw', data: event.data })
    }
  }
  return ws
}

export const fallbackData = {
  health: {
    status: 'ok',
    app: 'Child Emotion Detection Backend',
    version: '0.1.0',
    inference_mode: 'demo',
    timestamp: new Date().toISOString(),
  },
  cameras: [
    { id: 'camera-1', name: 'Demo Camera', source: 'demo', enabled: true, location: 'classroom' },
    { id: 'camera-2', name: '午休区', source: 'replay', enabled: true, location: 'nap room' },
  ],
  stats: {
    total_alerts: 4,
    active_alerts: 1,
    emotion_counts: { happy: 18, neutral: 24, sad: 7, fear: 5, anger: 4, surprise: 6 },
    recent_alerts: [],
  },
  alerts: [
    {
      id: 'demo-alert-1',
      camera_id: 'camera-1',
      track_id: 'person-1',
      emotion: 'sad',
      negative_score: 0.87,
      severity: 'critical',
      status: 'active',
      message: '检测到 person-1 高置信负面情绪: sad, score=0.87',
      created_at: new Date(Date.now() - 1000 * 60 * 4).toISOString(),
      updated_at: new Date().toISOString(),
      snapshot: { duration_seconds: 12, primary_confidence: 0.72 },
    },
    {
      id: 'demo-alert-2',
      camera_id: 'camera-2',
      track_id: 'person-3',
      emotion: 'fear',
      negative_score: 0.74,
      severity: 'warning',
      status: 'acknowledged',
      message: '午休区出现持续害怕表情，已确认',
      created_at: new Date(Date.now() - 1000 * 60 * 19).toISOString(),
      updated_at: new Date().toISOString(),
      snapshot: { duration_seconds: 18, primary_confidence: 0.66 },
    },
  ],
}
