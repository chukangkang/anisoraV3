# AnisoraV3 Video Generation API

基于 Wan2.1 模型的 OpenAI 兼容视频生成 API，支持 8 卡分布式推理。

## 功能特性

- ✅ **OpenAI 兼容接口** - 完全兼容 OpenAI 视频生成 API 规范
- ✅ **8卡分布式** - 支持多 GPU 分布式模型加载
- ✅ **异步任务** - 后台任务队列，支持状态查询，任务持久化
- ✅ **图像转视频(I2V)** - 支持从图像生成视频（单图/双图/三图）
- ✅ **多图权重控制** - 根据图片数量自动设置权重配置
- ✅ **360度角色旋转** - 支持360度角色旋转视频生成
- ✅ **提示词增强** - 自动追加质量评分参数

## 支持的参数

| 参数 | 可选值 | 默认值 |
|------|--------|--------|
| model | sora-2, sora-2-pro | sora-2 |
| size | 720x1280, 1280x720, 1024x1792, 1792x1024 | 720x1280 |
| seconds | 4, 5，8, 10 | 4 |
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

图像转视频(I2V)模式，需要提供输入图像。支持单图、双图和三图上传。

#### 单图上传（本地文件）

```bash
curl -X POST http://localhost:8000/v1/videos \
  -F "prompt=A cat playing piano on stage" \
  -F "size=720x1280" \
  -F "seconds=8" \
  -F "image=@/path/to/image.jpg"
```

#### 双图上传（首尾帧-本地文件）

```bash
curl -X POST http://localhost:8000/v1/videos \
  -F "prompt=A cat playing piano on stage" \
  -F "size=720x1280" \
  -F "seconds=8" \
  -F "image=@/path/to/image1.jpg" \
  -F "image1=@/path/to/image2.jpg"
```

#### 三图上传（中间帧插值-本地文件）

```bash
curl -X POST http://localhost:8000/v1/videos \
  -F "prompt=A cat playing piano on stage" \
  -F "size=720x1280" \
  -F "seconds=8" \
  -F "image=@/path/to/image1.jpg" \
  -F "image1=@/path/to/image2.jpg" \
  -F "image2=@/path/to/image3.jpg"
```

#### 单图上传（URL链接）

```bash
curl -X POST http://localhost:8000/v1/videos \
  -F "prompt=A cat playing piano on stage" \
  -F "size=720x1280" \
  -F "seconds=8" \
  -F "image_url=https://example.com/image.jpg"
```

#### 双图上传（URL链接）

```bash
curl -X POST http://localhost:8000/v1/videos \
  -F "prompt=A cat playing piano on stage" \
  -F "size=720x1280" \
  -F "seconds=8" \
  -F "image_url=https://example.com/image1.jpg" \
  -F "image_url1=https://example.com/image2.jpg"
```

#### 三图上传（URL链接）

```bash
curl -X POST http://localhost:8000/v1/videos \
  -F "prompt=A cat playing piano on stage" \
  -F "size=720x1280" \
  -F "seconds=8" \
  -F "image_url=https://example.com/image1.jpg" \
  -F "image_url1=https://example.com/image2.jpg" \
  -F "image_url2=https://example.com/image3.jpg"
```

> 注意：系统会自动追加质量评分参数 `aesthetic score: 5.5. motion score: 3.0. There is no text in the video.` 到提示词末尾。

### 创建视频任务 (URL链接方式)

如果图片托管在远程服务器，可以使用URL链接方式：

```bash
curl -X POST http://localhost:8000/v1/videos \
  -F "prompt=A cat playing piano on stage" \
  -F "size=720x1280" \
  -F "seconds=8" \
  -F "image_url=https://example.com/image.jpg"
```

### 创建360度旋转视频 (POST /v1/videos/360)

360度角色旋转功能，需要提供输入图像。

#### 本地文件方式

```bash
curl -X POST http://localhost:8000/v1/videos/360 \
  -F "prompt=A 360-degree turning and circling video of an anime character..." \
  -F "size=1280x720" \
  -F "seconds=5" \
  -F "image=@/path/to/image.jpg"
```

#### URL链接方式

```bash
curl -X POST http://localhost:8000/v1/videos/360 \
  -F "prompt=A 360-degree turning and circling video of an anime character..." \
  -F "size=1280x720" \
  -F "seconds=5" \
  -F "image_url=https://example.com/image.jpg"
```

> 注意：360度接口必须提供图片（本地文件或URL链接），推荐使用5秒时长以确保完整的360度旋转。

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

#### I2V图像转视频 (本地文件-单图)

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/videos",
    data={
        "prompt": "A cat playing piano on stage",
        "size": "720x1280",
        "seconds": "8",
    },
    files={"image": open("input.jpg", "rb")}
)

task = response.json()
print(f"Task ID: {task['id']}")
```

#### I2V图像转视频 (双图上传)

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/videos",
    data={
        "prompt": "A cat playing piano on stage",
        "size": "720x1280",
        "seconds": "8",
    },
    files={
        "image": open("image1.jpg", "rb"),
        "image1": open("image2.jpg", "rb")
    }
)

task = response.json()
print(f"Task ID: {task['id']}")
```

#### I2V图像转视频 (三图上传)

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/videos",
    data={
        "prompt": "A cat playing piano on stage",
        "size": "720x1280",
        "seconds": "8",
    },
    files={
        "image": open("image1.jpg", "rb"),
        "image1": open("image2.jpg", "rb"),
        "image2": open("image3.jpg", "rb")
    }
)

task = response.json()
print(f"Task ID: {task['id']}")
```

#### I2V图像转视频 (单图URL链接)

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/videos",
    data={
        "prompt": "A cat playing piano on stage",
        "size": "720x1280",
        "seconds": "8",
    },
    # URL链接方式
    data={"image_url": "https://example.com/image.jpg"}
)

task = response.json()
print(f"Task ID: {task['id']}")
```

#### I2V图像转视频 (双图URL链接)

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/videos",
    data={
        "prompt": "A cat playing piano on stage",
        "size": "720x1280",
        "seconds": "8",
    },
    data={
        "image_url": "https://example.com/image1.jpg",
        "image_url1": "https://example.com/image2.jpg"
    }
)

task = response.json()
print(f"Task ID: {task['id']}")
```

#### I2V图像转视频 (三图URL链接)

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/videos",
    data={
        "prompt": "A cat playing piano on stage",
        "size": "720x1280",
        "seconds": "8",
    },
    data={
        "image_url": "https://example.com/image1.jpg",
        "image_url1": "https://example.com/image2.jpg",
        "image_url2": "https://example.com/image3.jpg"
    }
)

task = response.json()
print(f"Task ID: {task['id']}")
```

#### 查询状态和下载

```python
import requests

# 查询状态
status_response = requests.get(f"http://localhost:8000/v1/videos/{task['id']}")
print(status_response.json())

# 下载视频
if status_response.json()['status'] == 'completed':
    video_response = requests.get(f"http://localhost:8000/v1/videos/{task['id']}/content")
    with open("output.mp4", "wb") as f:
        f.write(video_response.content)
```

#### 生成360度旋转视频

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/videos/360",
    data={
        "prompt": "A 360-degree turning and circling video of an anime character...",
        "size": "1280x720",
        "seconds": "5",
    },
    files={"image": open("character.png", "rb")}
)

task = response.json()
print(f"Task ID: {task['id']}")
print(f"Status: {task['status']}")
```
