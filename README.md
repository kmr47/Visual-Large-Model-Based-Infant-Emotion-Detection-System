# 基于视觉模型的幼儿情绪检测系统

本项目提供一个本地运行的幼儿情绪检测演示系统：Vue 3 前端通过 FastAPI 后端调用人脸检测与情绪推理，并提供检测事件、告警、儿童档案、匿名身份及日报/周报等管理界面。

> **隐私与使用边界**：请仅使用获得合法授权的数据和摄像头。不要将可识别幼儿身份的视频、图像、数据库或访问凭据提交到 GitHub。模型输出仅用于研究/演示，不应作为医疗诊断、儿童评价或单独作出教育决定的依据。部署前须配置强随机密钥并进行安全评估。

## 技术栈

- 后端：Python 3.10+、FastAPI、Uvicorn、SQLite、OpenCV、PyTorch
- 人脸检测：MediaPipe 模型文件（随代码提供）
- 情绪分类：ViT-B/16；训练权重需单独取得，不包含在此源码仓库中
- 前端：Node.js 18+、Vue 3、Vite

## 目录

```text
backend/      FastAPI 接口、业务逻辑和本地配置模板
frontend/     Vue 3 管理界面
scripts/      训练、评估和数据处理脚本
docs/         架构、实验和隐私功能说明
models/       模型加载目录；权重不随仓库分发
```

## 本地运行

### 1. 配置后端

在项目根目录运行：

```powershell
Copy-Item backend/.env.example backend/.env
```

编辑 `backend/.env`，将 `EMOTION_PRIVACY_SECRET` 和 `EMOTION_API_TOKEN` 分别替换为不同的强随机值。不要把该文件提交或发送给他人。`mock` 模式可用于验证界面和 API，不需要模型权重。

安装依赖并启动 API：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend/requirements.txt
python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8002
```

健康检查和 API 文档：

- http://127.0.0.1:8002/health
- http://127.0.0.1:8002/docs

首次启动时，SQLite 数据库会在 `backend/data/` 中创建；该目录已加入 Git 忽略规则。

### 2. 配置并启动前端

```powershell
Copy-Item frontend/.env.development.example frontend/.env.development
cd frontend
npm ci
npm run dev -- --host 127.0.0.1 --port 5174
```

浏览器打开 http://127.0.0.1:5174 。若后端运行在不同地址，请修改 `frontend/.env.development` 中的 `VITE_API_BASE`。

## 推理模式与模型权重

- `mock`：默认离线演示模式，不加载情绪模型。
- `local`：本地真实推理，需要兼容的 ViT-B/16 权重及对应 PyTorch/MediaPipe 依赖。
- `cloud`：调用自行管理的远程推理 API，需要配置 `EMOTION_CLOUD_INFERENCE_URL`。
- `replay`：需要本地实验预测 CSV 文件；数据文件不包含在仓库中。

由于模型权重体积较大且数据/授权条件需单独确认，仓库不提供 `.pth` 权重、数据集、视频或个人运行数据库。将获授权的模型权重放入项目根目录 `models/`，并按后端模型加载配置运行。不要上传私有训练数据或第三方受限数据。

## 安全说明

- `backend/.env` 和前端本地环境文件不得提交。
- API 管理令牌与匿名化密钥必须分别设置为独立随机值；不得沿用示例字符串。
- 服务默认绑定 `127.0.0.1`，仅供本机开发。公开部署前需配置 HTTPS、身份认证、访问控制、备份和数据保留策略。
- 数据库可能包含敏感事件信息；请妥善保管并按适用法律及机构要求删除。
- 运行前确认视频/图像中所有人员的授权和告知要求。

## 开发验证

前端生产构建：

```powershell
cd frontend
npm ci
npm run build
```

后端健康检查见上文。详细接口及功能说明请参阅 `docs/`。
