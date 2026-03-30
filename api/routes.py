# -*- coding: utf-8 -*-
"""
FastAPI路由模块 - OpenAI兼容的视频生成API
"""
import os
import time
import logging
from typing import Optional, List
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from pydantic import BaseModel

from .config import (
    API_CONFIG,
    MODEL_CONFIG,
    VideoModel,
    VideoSize,
    VideoSeconds,
    VideoQuality,
    VideoStatus,
    validate_openai_params,
)
from .converter import OpenAICreateVideoRequest
from .generator import get_generator, reset_generator
from .task_manager import get_task_manager, VideoTask


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)


# 创建FastAPI应用
app = FastAPI(
    title="AnisoraV3 Video Generation API",
    description="OpenAI兼容的视频生成API，基于Wan2.1模型",
    version="1.0.0",
)


# ==================== Pydantic模型 ====================

class CreateVideoRequest(BaseModel):
    """创建视频请求模型"""
    prompt: str
    model: Optional[str] = None
    size: Optional[str] = None
    seconds: Optional[str] = None
    quality: Optional[str] = None


class VideoResponse(BaseModel):
    """视频响应模型"""
    id: str
    object: str = "video"
    model: str
    status: str
    progress: int
    created_at: int
    size: str
    seconds: str
    quality: str


class VideoListResponse(BaseModel):
    """视频列表响应模型"""
    data: List[VideoResponse]
    first_id: Optional[str] = None
    last_id: Optional[str] = None
    has_more: bool = False


# ==================== 辅助函数 ====================

def task_to_response(task: VideoTask) -> VideoResponse:
    """将任务转换为API响应格式
    
    Args:
        task: 视频任务
        
    Returns:
        API响应对象
    """
    return VideoResponse(
        id=task.id,
        object="video",
        model=task.model,
        status=task.status,
        progress=task.progress,
        created_at=task.created_at,
        size=task.size,
        seconds=task.seconds,
        quality=task.quality,
    )


async def run_generation_task(
    task_id: str,
    prompt: str,
    image_data: Optional[bytes],
    size: str,
    seconds: str,
):
    """在后台运行视频生成任务

    Args:
        task_id: 任务ID
        prompt: 提示词
        image_data: 图像数据(可选)
        size: 分辨率
        seconds: 时长(秒)
    """

    from .converter import ParameterConverter

    task_manager = get_task_manager()
    generator = get_generator(
        checkpoint_dir=MODEL_CONFIG.checkpoint_dir,
        output_dir="/root/anisoraV3/output_videos_any",
        distributed=True,  # 8卡分布式
        world_size=8,
    )

    try:
        # 更新状态为处理中
        task_manager.update_task(task_id, status=VideoStatus.PROCESSING.value, progress=10)

        # 参数转换
        converter = ParameterConverter(checkpoint_dir=MODEL_CONFIG.checkpoint_dir)

        # 创建任务工作目录 (使用任务ID命名)
        project_root = "/root/anisoraV3"
        input_base_dir = os.path.join(project_root, "anisora_input")
        work_dir = os.path.join(input_base_dir, task_id)
        
        request = OpenAICreateVideoRequest(
            prompt=prompt,
            size=size,
            seconds=seconds,
        )

        if image_data:
            params = converter.convert_with_image(request, image_data, work_dir=work_dir)
        else:
            params = converter.convert(request)

        # 更新进度
        task_manager.update_task(task_id, progress=30)

        # 执行生成
        result = generator.generate(
            prompt=prompt,
            image_path=params.image if hasattr(params, 'image') else None,
            size=size,
            seconds=seconds,
            seed=params.base_seed,
            task_id=task_id,
        )

        # 更新结果
        if result["success"]:
            task_manager.update_task(
                task_id,
                status=VideoStatus.COMPLETED.value,
                progress=100,
                video_path=result["video_path"],
            )
            logger.info(f"Task {task_id} completed successfully")
        else:
            task_manager.update_task(
                task_id,
                status=VideoStatus.FAILED.value,
                error=result["error"],
            )
            logger.error(f"Task {task_id} failed: {result['error']}")

    except Exception as e:
        logger.error(f"Task {task_id} error: {e}")
        import traceback
        traceback.print_exc()

        task_manager.update_task(
            task_id,
            status=VideoStatus.FAILED.value,
            error=str(e),
        )


async def run_generation_task_360(
    task_id: str,
    prompt: str,
    image_data: Optional[bytes],
    size: str,
    seconds: str,
):
    """在后台运行360度旋转视频生成任务

    Args:
        task_id: 任务ID
        prompt: 提示词(包含图像路径格式: image_path@@prompt&&image_position)
        image_data: 图像数据(可选)
        size: 分辨率
        seconds: 时长(秒)
    """

    from .converter import ParameterConverter

    task_manager = get_task_manager()
    generator = get_generator(
        checkpoint_dir=MODEL_CONFIG.checkpoint_dir,
        output_dir="/root/anisoraV3/output_videos_360",
        distributed=True,  # 8卡分布式
        world_size=8,
    )

    try:
        # 更新状态为处理中
        task_manager.update_task(task_id, status=VideoStatus.PROCESSING.value, progress=10)

        # 参数转换 - 360度使用不同的转换器配置
        converter = ParameterConverter(checkpoint_dir=MODEL_CONFIG.checkpoint_dir)

        # 创建任务工作目录 (使用任务ID命名)
        project_root = "/root/anisoraV3"
        input_base_dir = os.path.join(project_root, "anisora_input")
        work_dir = os.path.join(input_base_dir, task_id)

        request = OpenAICreateVideoRequest(
            prompt=prompt,
            size=size,
            seconds=seconds,
        )

        # 360度模式：使用传入的image_data或从prompt解析
        if image_data:
            # 使用传入的图像数据
            params = converter.convert_with_image(request, image_data, work_dir=work_dir)
        elif "@@" in prompt:
            # 从prompt中提取图像路径 (格式: image_path@@prompt&&image_position)
            image_part = prompt.split("@@")[0]
            if os.path.exists(image_part):
                params = converter.convert(request)
                params.image = image_part
            else:
                params = converter.convert(request)
        else:
            params = converter.convert(request)

        # 更新进度
        task_manager.update_task(task_id, progress=30)

        # 执行360度生成 - 使用frame_num=81 (5秒@16fps)
        result = generator.generate_360(
            prompt=prompt,
            image_path=params.image if hasattr(params, 'image') and params.image else None,
            size=size,
            seconds=seconds,
            seed=params.base_seed,
            task_id=task_id,
            frame_num=81,  # 360度推荐81帧
        )

        # 更新结果
        if result["success"]:
            task_manager.update_task(
                task_id,
                status=VideoStatus.COMPLETED.value,
                progress=100,
                video_path=result["video_path"],
            )
            logger.info(f"Task {task_id} 360 completed successfully")
        else:
            task_manager.update_task(
                task_id,
                status=VideoStatus.FAILED.value,
                error=result["error"],
            )
            logger.error(f"Task {task_id} 360 failed: {result['error']}")

    except Exception as e:
        logger.error(f"Task {task_id} 360 error: {e}")
        import traceback
        traceback.print_exc()

        task_manager.update_task(
            task_id,
            status=VideoStatus.FAILED.value,
            error=str(e),
        )


@app.get("/v1/videos")
async def list_videos(limit: int = 20, before: Optional[str] = None):
     """列出视频(GET /v1/videos)
     
     OpenAI兼容接口 - 获取视频列表
     
     Args:
          limit: 返回数量限制(默认20)
          before: 分页游标(可选)
          
     Returns:
          视频列表响应  
     """
     
     task_manager = get_task_manager()
     tasks = task_manager.list_tasks(limit=limit)
     
     # 转换为响应格式  
     video_list = [task_to_response(task) for task in tasks]
     
     return VideoListResponse(
         data=video_list,
         first_id=video_list[0].id if video_list else None,
         last_id=video_list[-1].id if video_list else None,
         has_more=len(tasks) >= limit,
     )


@app.post("/v1/videos", response_model=VideoResponse)
async def create_video(
    background_tasks: BackgroundTasks,
    prompt: str = Form(...),
    model: Optional[str] = Form(None),
    size: Optional[str] = Form(None),
    seconds: Optional[str] = Form(None),
    quality: Optional[str] = Form(None),
    image: UploadFile = File(None),
    image_url: Optional[str] = Form(None),
):
    """创建视频(POST /v1/videos)

    OpenAI兼容接口 - 基于图像生成视频(I2V)

    支持两种图片输入方式:
    1. 本地文件上传 - 使用image参数
    2. URL链接 - 使用image_url参数

    Args:
        prompt: 文本提示词(必填)
        model: 模型名称
        size: 分辨率
        seconds: 时长(4, 8, 12)
        quality: 质量
        image: 本地图像文件(必填，与image_url二选一)
        image_url: 图像URL链接(必填，与image二选一)

    Returns:
        视频任务响应
    """

    # 参数验证 - 图片必须提供
    if not image and not image_url:
        raise HTTPException(
            status_code=400,
            detail="Image is required for video generation. Please provide either 'image' (local file) or 'image_url' (URL link)."
        )

    if image and image_url:
        raise HTTPException(
            status_code=400,
            detail="Please provide only one of 'image' or 'image_url', not both."
        )

    if not validate_openai_params(model=model, size=size, seconds=seconds):
        raise HTTPException(
            status_code=400,
            detail="Invalid parameter value"
        )

    # 设置默认值
    model = model or API_CONFIG.default_model
    size = size or API_CONFIG.default_size
    seconds = seconds or API_CONFIG.default_seconds

    # 处理图片数据 - 本地文件或URL
    image_data = None

    if image:
        # 本地文件上传
        image_data = await image.read()
    elif image_url:
        # URL链接下载
        try:
            import requests
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()
            image_data = response.content
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to download image from URL: {str(e)}"
            )

    # 创建任务
    task_manager = get_task_manager()

    task = task_manager.create_task(
        prompt=prompt,
        model=model,
        size=size,
        seconds=seconds,
        quality=quality or API_CONFIG.default_quality,
    )

    # 添加后台任务执行生成
    background_tasks.add_task(
        run_generation_task,
        task_id=task.id,
        prompt=prompt,
        image_data=image_data,
        size=size,
        seconds=seconds,
    )

    return task_to_response(task)


@app.post("/v1/videos/360", response_model=VideoResponse)
async def create_video_360(
    background_tasks: BackgroundTasks,
    prompt: str = Form(...),
    model: Optional[str] = Form(None),
    size: Optional[str] = Form(None),
    seconds: Optional[str] = Form(None),
    quality: Optional[str] = Form(None),
    image: UploadFile = File(None),
    image_url: Optional[str] = Form(None),
):
    """创建360度旋转视频(POST /v1/videos/360)

    360-Degree Character Rotation - 基于输入图像生成360度旋转视频

    支持两种图片输入方式:
    1. 本地文件上传 - 使用image参数
    2. URL链接 - 使用image_url参数

    Args:
        prompt: 文本提示词(必填)
        model: 模型名称
        size: 分辨率(推荐1280x720)
        seconds: 时长(推荐5秒确保完整360度旋转)
        quality: 质量
        image: 本地图像文件(可选，与image_url二选一)
        image_url: 图像URL链接(可选，与image二选一)

    Returns:
        视频任务响应
    """

    # 参数验证 - 图片必须提供
    if not image and not image_url:
        raise HTTPException(
            status_code=400,
            detail="Image is required for 360-degree video generation. Please provide either 'image' (local file) or 'image_url' (URL link)."
        )

    if image and image_url:
        raise HTTPException(
            status_code=400,
            detail="Please provide only one of 'image' or 'image_url', not both."
        )

    if not validate_openai_params(model=model, size=size, seconds=seconds):
        raise HTTPException(
            status_code=400,
            detail="Invalid parameter value"
        )

    # 设置默认值 - 360度推荐5秒
    model = model or API_CONFIG.default_model
    size = size or "1280x720"
    seconds = seconds or "5"

    # 创建任务
    task_manager = get_task_manager()

    task = task_manager.create_task(
        prompt=prompt,
        model=model,
        size=size,
        seconds=seconds,
        quality=quality or API_CONFIG.default_quality,
        video_type="360",  # 标记为360度旋转视频
    )

    # 处理图片数据 - 本地文件或URL
    image_data = None

    if image:
        # 本地文件上传
        image_data = await image.read()
    elif image_url:
        # URL链接下载
        try:
            import requests
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()
            image_data = response.content
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to download image from URL: {str(e)}"
            )

    # 添加后台任务执行360度生成
    background_tasks.add_task(
        run_generation_task_360,
        task_id=task.id,
        prompt=prompt,
        image_data=image_data,
        size=size,
        seconds=seconds,
    )

    return task_to_response(task)


@app.get("/v1/videos/{video_id}", response_model=VideoResponse)
async def retrieve_video(video_id: str):
      """获取视频(GET /v1/videos/{video_id})
      
      OpenAI兼容接口 - 获取特定视频的状态和信息
      
      Args:
           video_id: 视频ID  
           
      Returns:
           视频响应  
      """
      
      task_manager = get_task_manager()
      task = task_manager.get_task(video_id)
      
      if not task:
          raise HTTPException(status_code=404, detail="Video not found")
      
      return task_to_response(task)


@app.delete("/v1/videos/{video_id}")
async def delete_video(video_id: str):
      """删除视频(DELETE /v1/videos/{video_id})
      
      OpenAI兼容接口 - 删除指定的视频任务
      
      Args:
           video_id: 视频ID  
           
      Returns:
           删除结果  
      """
      
      task_manager = get_task_manager()
      
      # 检查任务是否存在  
      task = task_manager.get_task(video_id)
      if not task:
          raise HTTPException(status_code=404, detail="Video not found")
      
      # 如果有视频文件，删除它  
      if task.video_path and os.path.exists(task.video_path):
          try:
              os.remove(task.video_path)
              logger.info(f"Deleted video file: {task.video_path}")
          except Exception as e:
              logger.warning(f"Failed to delete video file {task.video_path}: {e}")
      
      # 删除任务记录  
      success = task_manager.delete_task(video_id)
      
      if success:
          return {"deleted": video_id, "object": "video"}
      else:
          raise HTTPException(status_code=500, detail="Failed to delete video")


@app.get("/v1/videos/{video_id}/content")
async def download_video_content(video_id: str):
    """下载视频内容(GET /v1/videos/{video_id}/content)

    OpenAI兼容接口 - 下载生成的视频文件

    Args:
        video_id: 视频ID

    Returns:
        视频文件流
    """

    task_manager = get_task_manager()
    task = task_manager.get_task(video_id)

    if not task:
        raise HTTPException(status_code=404, detail="Video not found")

    if task.status != VideoStatus.COMPLETED.value:
        raise HTTPException(
            status_code=400,
            detail=f"Video is not ready (status: {task.status})"
        )

    if not task.video_path or not os.path.exists(task.video_path):
        raise HTTPException(status_code=404, detail="Video file not found")

    # 返回文件流
    return FileResponse(
        path=task.video_path,
        media_type="video/mp4",
        filename=f"{video_id}.mp4",
    )


# 健康检查端点 
@app.get("/health")
async def health_check():
      """健康检查"""
      return {"status": "healthy", "timestamp": int(time.time())}


# 启动和关闭事件 
@app.on_event("startup")
async def startup_event():
      """应用启动事件"""
      logger.info("Starting AnisoraV3 Video Generation API...")
      
      
@app.on_event("shutdown") 
async def shutdown_event():
      """应用关闭事件"""
      logger.info("Shutting down AnisoraV3 Video Generation API...")
      
      # 清理资源  
      reset_generator()
