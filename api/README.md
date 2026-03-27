# AnisoraV3 Video Generation API

基于 Wan2.1 模型的 OpenAI 兼容视频生成 API，支持 8 卡分布式推理。

## 功能特性

- ✅ **OpenAI 兼容接口** - 完全兼容 OpenAI 视频生成 API 规范
- ✅ **8卡分布式** - 支持多 GPU 分布式模型加载
- ✅ **异步任务** - 后台任务队列，支持状态查询
- ✅ **图像转视频(I2V)** - 支持从图像生成视频

## 支持的参数

| 参数 | 可选值 | 默认值 |
|------|--------|--------|
| model | sora-2, sora-2-pro | sora-2 |
| size | 720x1280, 1280x720, 1024x1792, 1792x1024 | 720x1280 |
| seconds | 4, 8, 12 | 4 |
| quality | standard, high | standard |

## 安装依赖

```bash
pip install -r requirements.txt
```

## 启动服务

```bash
python run_api.py
```

服务将在 `http://localhost:8000` 启动。

## API 接口

### 创建视频任务 (POST /v1/videos)

```bash
curl -X POST http://localhost:8000/v1/videos \
  -F "prompt=A cat playing piano on stage" \
  -F "size=720x1280" \
  -F "seconds=8"
```

### 带图像创建视频 (POST /v1/videos/generation)

```bash
curl -X POST http://localhost:8000/v1/videos/generation \
  -F "prompt=A cat playing piano on stage" \
  -F "size=720x1280" \
  -F "seconds=8" \
  -F "image=@/path/to/image.jpg"
```

### 获取视频状态 (GET /v1/videos/{video_id})

```bash
curl http://localhost:8000/v1/videos/video_123abc
```

### 下载视频 (GET /v1/videos/{video_id}/content)

```bash
curl -o output.mp4 http://localhost:8000/v1/videos/video_123abc/content
```

### 删除视频 (DELETE /v1/videos/{video_id})

```bash
curl -X DELETE http://localhost:8000/v1/videos/video_123abc
```

### 列出所有视频 (GET /v1/videos)

```bash
curl http://localhost:8000/v1/videos?limit=20
```

## 项目结构

```
api/
├── __init__.py          # 包导出
├── config.py           # 配置模块
├── converter.py        # 参数转换器
├── generator.py        # 视频生成器
├── model_loader.py     # 模型加载器(分布式)
├── routes.py           # FastAPI路由
└── task_manager.py     # 任务管理器

run_api.py              # 服务入口
```

## 配置说明

在 `api/config.py` 中可以修改:

- `MODEL_CONFIG.checkpoint_dir` - 模型检查点目录
- `DISTRIBUTED_CONFIG.world_size` - GPU数量(默认8)
- `API_CONFIG.port` - 服务端口(默认8000)

## 使用示例

### Python 调用示例

```python
import requests

# 创建视频任务
response = requests.post(
    "http://localhost:8000/v1/videos/generation",
    data={
        "prompt": "A cat playing piano on stage",
        "size": "720x1280",
        "seconds": "8",
    },
    files={"image": open("input.jpg", "rb")}
)

task = response.json()
print(f"Task ID: {task['id']}")
print(f"Status: {task['status']}")

# 查询状态
status_response = requests.get(f"http://localhost:8000/v1/videos/{task['id']}")
print(status_response.json())

# 下载视频
if status_response.json()['status'] == 'completed':
    video_response = requests.get(f"http://localhost:8000/v1/videos/{task['id']}/content")
    with open("output.mp4", "wb") as f:
        f.write(video_response.content)
```
