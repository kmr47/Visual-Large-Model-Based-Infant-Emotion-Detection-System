# v0.3 新增功能说明：隐私优先数据库 + 情绪日报/周报 + 匿名身份 re-ID

> 本目录是原工程的开发副本（`emotion-detector-v2-report`），用于新增并测试
> "情绪日报/周报 + 隐私保护 + 匿名身份" 功能。原始工程与备份均未改动：
> - 原始工程：`emotion-detector/`
> - 完整备份：`emotion-detector-backup-20260902/`（1.24GB，含模型权重与数据）

---

## 一、新增内容一览

| 层 | 文件 | 说明 |
|---|---|---|
| 后端 | `backend/app/db.py` | SQLite 持久化层：建表、匿名事件写入、时间窗查询、过期清理、身份库 |
| 后端 | `backend/app/security.py` | 可选 Bearer Token 访问控制（常量时间比较） |
| 后端 | `backend/app/routers/reports.py` | 日报/周报聚合 + 隐私状态 + 管理接口 |
| 后端 | `backend/app/state.py` | 告警/情绪计数落库；启动时从 DB 恢复历史告警 |
| 后端 | `backend/app/routers/inference.py` | 图片逐人脸、视频 2s/人抽样写入匿名事件（带最终代号） |
| 后端 | `backend/app/routers/identities.py` | 匿名身份库、待确认、认领/解除/删除特征原型 |
| 后端 | `backend/app/services/identity_service.py` | 特征匹配/聚合/三档阈值决策（无模型依赖） |
| 后端 | `backend/app/services/emotion_service.py` | local 推理同时提取 768 维身份特征；视频轨道级决策 |
| 后端 | `backend/app/main.py` | lifespan 初始化 DB + 后台保留期清理任务 |
| 前端 | `frontend/src/components/ReportView.vue` | 日报/周报页 + 检测记录/认领 + 匿名身份面板 |
| 前端 | `frontend/src/App.vue` | 侧栏「情绪报表」；工作台/告警显示匿名代号 |
| 前端 | `frontend/src/services/api.js` | 报表/privacy/identities/认领 API 封装 |
| 实验 | `scripts/vit_embedding_spike.py` | 特征稳定性离线实验（可复现） |
| 实验 | `scripts/reid_backbone_compare.py`、`reid_segment_compare.py` | 特征 backbone 与轨道聚合对比 |

## 二、数据库 Schema（隐私优先设计）

三张业务表 + 元数据表，**不含任何图像/视频/文件名/人脸框**：

```
children:       id, nickname, class_name, note, enabled, created_at
                （教师维护的“假名 ↔ 昵称”档案，不采集人脸）
emotion_events: id, event_uuid, child_id(NULL=匿名), child_anon_id, camera_id, source,
                emotion, confidence, negative_score, alert_triggered,
                event_request_id, track_label, created_at
                （event_request_id + track_label 用于“认领”，不涉及面容）
alert_events:   id, child_anon_id, camera_id, emotion, negative_score,
                severity, status, message, duration_seconds, created_at, updated_at
system_meta:    key, value
```

### 隐私设计落点（对应论文第 6 章 / 伦理讨论）

1. **数据最小化——只留"情绪事件"，不留"脸"**
   - 落库字段只有情绪标签、置信度、匿名ID、时间戳等统计所需字段；
   - 图片字节、文件名、bbox 一律不写库（`db.py` 无对应列，代码注释声明）；
   - `privacy` 接口可直接向评审展示 `stores_images: false`。

2. **匿名化 + 假名化**
   - `child_anon_id = HMAC-SHA256(camera_id|track_id, EMOTION_PRIVACY_SECRET)` 截断；
   - 单向不可逆：同一孩子跨帧稳定（可做个体趋势），但无密钥无法反推身份；
   - 更换密钥后所有假名失效，等价于"重新匿名化"。

3. **留存期限 + 自动清理**
   - `EMOTION_RETENTION_DAYS`（默认 30 天）；
   - `main.py` lifespan 启动后台任务，每 `EMOTION_PURGE_INTERVAL_SECONDS` 清理过期行；
   - 论文可写"数据生命周期管理"。

4. **访问控制（可选，已实现）**
   - 配置 `EMOTION_API_TOKEN` 后：
     - `POST /api/reports/purge`、`POST /api/reports/seed`、children 写操作需要
       `Authorization: Bearer <token>`（未带/错误 → 401）；
     - 只读接口保持开放，保证演示 UI 流畅；
   - 前端报表页的「数据管理」区支持填写口令后执行清除/生成。

5. **"认领"式个体关联（方案 A，非生物识别）**
   - 检测流水线只产出匿名事件；教师登记档案（昵称/班级）后，可把某次推理的
     某个轨道（request_id + track_label）"认领"给该孩子；
   - 不采集人脸、不做人脸识别，认领是**行为关联**（教师凭肉眼判断画面中
     轨道对应哪个孩子）；
   - 删除档案 → 全部关联事件自动回退为匿名（`delete_child` 先置 NULL 再删行）。

## 三、报表 API

| 接口 | 说明 |
|---|---|
| `GET /api/reports/daily?date=YYYY-MM-DD` | 日报：情绪分布、负面统计、24h 时段直方图、涉及幼儿数、环比前一日 |
| `GET /api/reports/weekly?start=YYYY-MM-DD` | 周报（周一起始）：逐日序列 + 周聚合 + 环比上周 |
| `GET /api/reports/privacy` | 隐私姿态与存储统计（答辩演示用） |
| `POST /api/reports/purge` | 清除全部事件（需 token，GDPR 式擦除演示） |
| `POST /api/reports/seed?days&per_day` | 生成带 `source=demo` 标记的演示数据（需 token；存在档案时自动按孩子分布） |
| `GET /api/children` | 孩子档案列表（含事件数） |
| `POST /api/children` | 登记孩子（需 token） |
| `DELETE /api/children/{id}` | 删除档案（关联事件自动匿名化，需 token） |
| `POST /api/children/claim` | 认领：把某次推理某轨道的事件关联给孩子（需 token） |
| `POST /api/children/{id}/unclaim` | 解除认领（回匿名，需 token） |
| `GET /api/children/{id}/daily?date=` | 该孩子个人日报（仅统计已认领事件） |

时区说明：聚合按 **UTC 日界**，接口返回 `tz_note` 字段注明。

## 四、如何启动（副本）

```powershell
# 后端（8002 端口，避免与原始版 8001 冲突）
cd emotion-detector-v2-report\backend
# 推理模式在 backend\.env 中配置：
#   EMOTION_INFERENCE_MODE=mock   → 确定性演示（不加载模型，最快）
#   EMOTION_INFERENCE_MODE=local  → 真实本地推理（默认已切为 local）
#                                   MediaPipe 人脸检测 + 本地 ViT 情绪分类，
#                                   权重优先 calibrated → fixed → age5
python -m uvicorn app.main:app --host 127.0.0.1 --port 8002

# 前端（5174 端口）
cd emotion-detector-v2-report\frontend
$env:VITE_API_BASE = "http://127.0.0.1:8002"
npm run dev -- --port 5174
```

浏览器打开 `http://127.0.0.1:5174` → 左侧栏「报表」图标进入日报/周报页。

> 说明：`local` 模式图片按“检测到的人脸”逐个跑 ViT（缓存后约 0.1–0.5s/张）；
> 视频按 0.6s/人间隔分类，首次请求含模型加载约 10–20s，后续正常。
> 模型权重：`models/vit_b16_age5_plus_age10_calibrated_best.pth`（优先）等三个 ViT-B/16。

## 五、匿名身份（跨视频 re-ID，v0.3）

**目标**：解决「孩子离开镜头一段时间再回来，系统还认得出吗」的问题。
旧版匿名代号由 `HMAC(camera, person-N)` 派生，`person-N` 是一次视频内的临时轨道号，
跨视频会重置——同一孩子每次上传都是新代号。v0.3 改为基于**人脸特征原型的匿名重识别**：

### 原理

```
检测到人脸 crop
  → ImageNet 预训练 ViT-B/16（与情绪模型解耦）提取 768 维归一化特征
  → 同一视频轨道内多帧特征取均值（轨道级聚合，轨道内相似度实测 ~0.99）
  → 与身份库中所有原型求余弦相似度，按阈值三档决策：
       sim >= 0.80       自动复用已有匿名代号（跨视频稳定）
       0.62 <= sim < 0.80  生成一条「轨道级待确认」（教师可选合并/保留/忽略，48h 自动过期）
       sim < 0.62       自动创建新的匿名身份
```

### 设计要点

- **数据最小化**：身份库存的是不可逆的 768 维特征原型（约 3KB/人），不存人脸图像、
  视频、文件名、边框；特征原型随事件一起受留存期约束，到期自动删除。
- **宁碎片化、不误认**：灰色带一律待确认，绝不把两个孩子自动合并；碎片化身份可由
  教师事后一键合并。
- **默认匿名、教师认领**：认领一个匿名身份后，它之后每次出现自动归入该孩子个人日报；
  解除/删除原型即回到匿名（符合被遗忘权）。
- **实时链路**：视频轨道收集到 3 个特征样本后立即决策，之后的告警/事件都带最终代号；
  轨道离开画面时其决策被保留，不丢失。

### 离线实验（scripts/ 可复现）

| 实验 | 结果 |
|---|---|
| 同图轻微扰动（128 张图 512 视图）| 余弦相似度均值 0.985、P05 0.947（特征稳定）|
| 不同人脸两两比较（合成测试视频 3 段）| emotion backbone 0.21–0.43；imagenet backbone 0.37–0.53（分离有限，故阈值保守）|
| 同一孩子三段重入（原图/亮度变化/镜像）| 0.998 / 0.953 / 0.854，全部自动复用同一代号 |
| 同视频两次上传（跨视频稳定）| 6 个轨道全部 match 同一代号，零待确认 |

**诚实边界**：ViT 倒数第二层特征不是专为人脸识别设计的；真实部署中同人跨表情/光照的
相似度分布需要本校授权数据做 ROC 校准。当前阈值按「宁碎片化」取向保守设置。

### 相关接口

- `GET /api/identities`、`GET /api/identities/pending`（不返回特征向量）
- `POST /api/identities/pending/{id}/resolve`（link | keep | dismiss，需口令）
- `POST /api/identities/{anon}/claim | unclaim`（认领/解除身份，需口令）
- `DELETE /api/identities/{anon}`（删除特征原型，事件仍匿名）
- 推理响应新增 `track_identity_decisions`（match/new/pending + 相似度，不含向量）

## 五、会话级认领缩略图（v0.4，方案 B）

**背景**：匿名代号解决不了「老师怎么知道 c-xxxx 是哪个孩子」。认领必须看图认人，
但隐私设计禁止持久化图像。于是引入**会话级缩略图缓冲（认领即焚）**：

- 推理时为每条轨道截一张最佳人脸缩略图（96px JPEG，几 KB）；
- **只存进程内存**（`app/thumb_buffer.py`），不写 SQLite、不落盘、重启即失；
- 检测记录面板每条轨道显示代号 + 缩略脸，老师凭脸认领；
- **认领即焚**：认领成功后立即销毁该轨道缩略图；可手动焚毁整个批次；
- TTL 默认 2 小时自动过期，LRU 上限 300 张（`EMOTION_THUMB_*` 可调）；
- 读取端点在部署说明中应置于访问控制之后；销毁/清除需管理口令。

隐私叙事（论文可用）：数据库依旧零图像；图像只在内存中短暂存在，用于支撑一次
人工决策，决策完成即销毁——「采集最小化 + 短暂用途限制 + 立即删除」的完整链条。
相关接口：`GET /api/thumbs/{request_id}/{track_label}`（image/jpeg）、
`DELETE /api/thumbs/{request_id}[/{track_label}]`、`POST /api/thumbs/purge`、
`GET /api/thumbs/stats`；`/api/reports/privacy` 新增 `session_thumbnails` 状态。

## 六、演示动线建议（答辩）

1. 打开工作台 → 上传几张图片/一个视频 → 观察实时面板；
2. 切到「情绪报表」→ 今日日报出现真实事件统计；
3. 展开「数据管理」→ 输入管理口令（`thesis-demo-token-2026`）→ 生成近 7 天演示数据
   （`source=demo`）→ 查看周报逐日序列与环比；
4. 「导出 JSON」保存日报文件；
5. 隐私说明：报表页顶部徽标显示"匿名化 · 不存人脸图像"，必要时调 `/api/reports/privacy`
   展示 `stores_images: false` 与保留期配置。

## 七、已知说明

- 演示数据带有 `source=demo`，日报的 `by_source` 字段可区分；
- 视频推理按每 2 秒/人抽样落库，避免单条视频产生上千行冗余事件；
- 启动日志会打印 DB 路径与恢复的告警数，便于论文截图；
- re-ID 演示视频：`data/reid_demo_same_child.mp4`（同一孩子三段出现，中间黑屏间隔），
  上传后三段自动归入同一匿名代号，可直接演示「离开再回来」；
- 缩略图演示动线：上传视频 → 报表「检测记录/认领」→ 看到轨道缩略脸 → 认领为档案孩子
  → 缩略图立即焚毁（刷新后不再显示）→ `GET /api/thumbs/stats` 显示 entries=0；
- 身份特征模型为 ImageNet ViT-B/16（`EMOTION_REID_BACKBONE=imagenet`），与情绪模型
  分两次前向；如需单模型可切 `emotion`（跨表情稳定性更差，见上表）；
- 前提假设：论文应声明课堂视频处理已获园方与家长知情同意（缩略图短暂显示依赖此授权）。
