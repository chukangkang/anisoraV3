# -*- coding: utf-8 -*-
"""
FastAPI路由模块 - OpenAI兼容的视频生成API
"""
import os
import time
import asyncio
import logging
from typing import Optional, List, Union
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Union
from pathlib import Path

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
    images_data: Optional[List[bytes]] = None,
    weight_config: str = "&&1",
    size: str = "720x1280",
    seconds: str = "4",
):
    """在后台运行视频生成任务 (在独立线程中执行避免阻塞事件循环)

    Args:
        task_id: 任务ID
        prompt: 提示词
        images_data: 图片数据列表(可选)
        weight_config: 图片权重配置，如"&&1"或"&&1,1"
        size: 分辨率
        seconds: 时长(秒)
    """
    # 在线程池中执行耗时的同步生成任务，避免阻塞事件循环
    await asyncio.to_thread(_run_generation_task_sync, task_id, prompt, images_data, weight_config, size, seconds)


def _run_generation_task_sync(
    task_id: str,
    prompt: str,
    images_data: Optional[List[bytes]] = None,
    weight_config: str = "&&1",
    size: str = "720x1280",
    seconds: str = "4",
):
    """同步执行视频生成任务 (在线程池中运行)

    Args:
        task_id: 任务ID
        prompt: 提示词
        images_data: 图片数据列表(可选)
        weight_config: 图片权重配置
        size: 分辨率
        seconds: 时长(秒)
    """
    import asyncio
    from .converter import ParameterConverter

    # 创建新的事件循环来运行异步代码
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        # 获取task_manager和generator (在线程中)
        task_manager = get_task_manager()
        
        # 使用同步方式获取generator (避免在同步函数中调用async get_generator)
        from .generator import VideoGenerator
        generator = VideoGenerator(
            checkpoint_dir=MODEL_CONFIG.checkpoint_dir,
            output_dir="/root/anisoraV3/output_videos_any",
            distributed=True,
            world_size=8,
        )

        # 更新状态为处理中 - 需要在线程中创建协程并运行
        loop.run_until_complete(
            task_manager.update_task(task_id, status=VideoStatus.PROCESSING.value, progress=10)
        )

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

        # 根据图片数量选择转换方法
        if images_data and len(images_data) > 0:
            params = converter.convert_with_images(
                request, 
                images_data, 
                weight_config=weight_config,
                work_dir=work_dir
            )
        else:
            params = converter.convert(request)

        # 更新进度
        loop.run_until_complete(task_manager.update_task(task_id, progress=30))

        # 执行生成 (这是耗时的同步操作，会在线程中阻塞但不会影响主事件循环)
        result = generator.generate(
            prompt=params.prompt,  # 使用增强后的prompt（包含权重配置）
            image_path=None,      # 多图模式下在prompt中指定
            size=size,
            seconds=seconds,
            seed=params.base_seed,
            task_id=task_id,
        )

        # 更新结果
        if result["success"]:
            loop.run_until_complete(
                task_manager.update_task(
                    task_id,
                    status=VideoStatus.COMPLETED.value,
                    progress=100,
                    video_path=result["video_path"],
                )
            )
            logger.info(f"Task {task_id} completed successfully")
        else:
            loop.run_until_complete(
                task_manager.update_task(
                    task_id,
                    status=VideoStatus.FAILED.value,
                    error=result["error"],
                )
            )
            logger.error(f"Task {task_id} failed: {result['error']}")

    except Exception as e:
        logger.error(f"Task {task_id} error: {e}")
        import traceback
        traceback.print_exc()

        loop.run_until_complete(
            task_manager.update_task(
                task_id,
                status=VideoStatus.FAILED.value,
                error=str(e),
            )
        )
    finally:
        loop.close()


async def run_generation_task_360(
    task_id: str,
    prompt: str,
    image_data: Optional[bytearray],
    size: str,
    seconds: str,
):
    """在后台运行360度旋转视频生成任务 (在独立线程中执行避免阻塞事件循环)

    Args:
        task_id: 任务ID
        prompt: 提示词(包含图像路径格式: image_path@@prompt&&image_position)
        image_data: 图像数据(可选)
        size: 分辨率
        seconds: 时长(秒)
    """
    # 在线程池中执行耗时的同步生成任务，避免阻塞事件循环
    await asyncio.to_thread(_run_generation_task_360_sync, task_id, prompt, image_data, size, seconds)


def _run_generation_task_360_sync(
    task_id: str,
    prompt: str,
    image_data: Optional[bytearray],
    size: str,
    seconds: str,
):
    """同步执行360度旋转视频生成任务 (在线程池中运行)

    Args:
        task_id: 任务ID
        prompt: 提示词(包含图像路径格式: image_path@@prompt&&image_position)
        image_data: 图像数据(可选)
        size: 分辨率
        seconds: 时长(秒)
    """
    import asyncio
    from .converter import ParameterConverter

    # 创建新的事件循环来运行异步代码
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        # 获取task_manager和generator (在线程中)
        task_manager = get_task_manager()
        
        # 使用同步方式获取generator (避免在同步函数中调用async get_generator)
        from .generator import VideoGenerator
        generator = VideoGenerator(
            checkpoint_dir=MODEL_CONFIG.checkpoint_dir,
            output_dir="/root/anisoraV3/output_videos_360",
            distributed=True,
            world_size=8,
        )

        # 更新状态为处理中 - 需要在线程中创建协程并运行
        loop.run_until_complete(
            task_manager.update_task(task_id, status=VideoStatus.PROCESSING.value, progress=10)
        )

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
            params = converter.convert_with_image(request, bytes(image_data), work_dir=work_dir)
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
        loop.run_until_complete(task_manager.update_task(task_id, progress=30))

        # 执行360度生成 (这是耗时的同步操作，会在线程中阻塞但不会影响主事件循环)
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
            loop.run_until_complete(
                task_manager.update_task(
                    task_id,
                    status=VideoStatus.COMPLETED.value,
                    progress=100,
                    video_path=result["video_path"],
                )
            )
            logger.info(f"Task {task_id} 360 completed successfully")
        else:
            loop.run_until_complete(
                task_manager.update_task(
                    task_id,
                    status=VideoStatus.FAILED.value,
                    error=result["error"],
                )
            )
            logger.error(f"Task {task_id} 360 failed: {result['error']}")

    except Exception as e:
        logger.error(f"Task {task_id} 360 error: {e}")
        import traceback
        traceback.print_exc()

        loop.run_until_complete(
            task_manager.update_task(
                task_id,
                status=VideoStatus.FAILED.value,
                error=str(e),
            )
        )
    finally:
        loop.close()


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
     tasks = await task_manager.list_tasks(limit=limit)
     
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
    request: Request,
    prompt: str = Form(None),
    model: Optional[str] = Form(None),
    size: Optional[str] = Form(None),
    seconds: Optional[str] = Form(None),
    quality: Optional[str] = Form(None),
    image: UploadFile = File(default=None),
    image_url: Optional[str] = Form(default=None),
    image_url1: Optional[str] = Form(default=None),
    image_url2: Optional[str] = Form(default=None),
):
    """创建视频(POST /v1/videos)

    OpenAI兼容接口 - 基于图像生成视频(I2V)

    支持两种请求格式:
    1. multipart/form-data: 本地文件或URL链接
       - 本地文件: image, image1, image2 (最多3张)
       - URL链接: image_url, image_url1, image_url2 (最多3张)
    2. application/json: URL链接或Base64
       - image, image1, image2 (最多3张)

    多图权重配置:
    - 单图：权重为1.0
    - 双图：权重各0.5  
    - 三图：权重0.3, 0.7, 1.0

    Args:
        prompt: 文本提示词(必填)
        model: 模型名称
        size: 分辨率
        seconds: 时长(4, 8, 12)
        quality: 质量
        image: 第一张图像文件/URL/base64(必填)
        image1: 第二张图像URL/base64(可选)
        image2: 第三张图像URL/base64(可选)

    Returns:
        视频任务响应
    """

    # 检测请求类型并提取参数
    content_type = request.headers.get("content-type", "")
    
    # JSON格式请求
    if "application/json" in content_type or prompt is None:
        try:
            body = await request.json()
            prompt = body.get("prompt")
            model = body.get("model")
            size = body.get("size")
            seconds = body.get("seconds")
            quality = body.get("quality")
            json_image = body.get("image")
            json_image1 = body.get("image1")
            json_image2 = body.get("image2")
            
            # JSON模式：图片只能是URL或Base64
            has_json_image = any([json_image, json_image1, json_image2])
            
            if not has_json_image:
                raise HTTPException(
                    status_code=400,
                    detail="At least one image is required for video generation."
                )
            
            # 处理JSON图片数据
            from .converter import get_image_weight_config
            
            images_data_list = []
            url_list = [url for url in [json_image, json_image1, json_image2] if url]
            
            for url_or_base64 in url_list:
                try:
                    # 判断是URL还是Base64
                    if url_or_base64.startswith('data:image') or len(url_or_base64) > 200:
                        import base64
                        if ',' in url_or_base64:
                            base64_data = url_or_base64.split(',', 1)[1]
                        else:
                            base64_data = url_or_base64
                        images_data_list.append(base64.b64decode(base64_data))
                    else:
                        import requests as req_lib
                        response = req_lib.get(url_or_base64, timeout=30)
                        response.raise_for_status()
                        images_data_list.append(response.content)
                except Exception as e:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Failed to process image: {str(e)}"
                    )
            
            weight_config = get_image_weight_config(len(images_data_list))
            
            logger.info(f"JSON mode: Processing {len(images_data_list)} images with weight config: {weight_config}")
            
            # 设置默认值
            model = model or API_CONFIG.default_model
            size = size or API_CONFIG.default_size
            seconds = seconds or API_CONFIG.default_seconds
            
            if not validate_openai_params(model=model, size=size, seconds=seconds):
                raise HTTPException(
                    status_code=400,
                    detail="Invalid parameter value"
                )
            
            # 创建任务
            task_manager = get_task_manager()
            
            task = await task_manager.create_task(
                prompt=prompt,
                model=model,
                size=size,
                seconds=seconds,
                quality=quality or API_CONFIG.default_quality,
            )
            
            background_tasks.add_task(
                run_generation_task,
                task_id=task.id,
                prompt=prompt,
                images_data=images_data_list,
                weight_config=weight_config,
                size=size,
                seconds=seconds,
            )
            
            return task_to_response(task)
            
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid JSON request: {str(e)}"
            )
    
    # Form表单格式请求 (原有逻辑)
    # 参数验证 - 必须提供至少一张图片（本地文件或URL）
    has_local_image = image is not None
    has_url_image = any([image_url, image_url1, image_url2])
    
    if not has_local_image and not has_url_image:
        raise HTTPException(
            status_code=400,
            detail="At least one image is required for video generation."
        )

    if has_local_image and has_url_image:
        raise HTTPException(
            status_code=400,
            detail="Please provide either local files or URL links, not both."
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


    # 处理多张图片数据并获取权重配置
    from .converter import get_image_weight_config
    
    images_data_list = []
    
    if has_local_image:
        # 本地文件模式 - 收集所有上传的图片
        for img_file in [image]:
            if img_file is not None:
                img_data = await img_file.read()
                images_data_list.append(img_data)
    else:
        # URL链接模式 - 收集所有URL链接
        url_list = [url for url in [image_url, image_url1, image_url2] if url]
        
        for url in url_list:
            try:
                import requests
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                images_data_list.append(response.content)
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to download image from URL: {str(e)}"
                )
    
    # 根据图片数量获取权重配置
    weight_config = get_image_weight_config(len(images_data_list))
    
    logger.info(f"Processing {len(images_data_list)} images with weight config: {weight_config}")

    # 创建任务
    task_manager = get_task_manager()

    task = await task_manager.create_task(
        prompt=prompt,
        model=model,
        size=size,
        seconds=seconds,
        quality=quality or API_CONFIG.default_quality,
    )

    # 添加后台任务执行生成（传入图片列表和权重配置）
    background_tasks.add_task(
        run_generation_task,
        task_id=task.id,
        prompt=prompt,
        images_data=images_data_list,  # 改为图片列表
        weight_config=weight_config,   # 传入权重配置
        size=size,
        seconds=seconds,
    )

    return task_to_response(task)


@app.post("/v1/videos/360", response_model=VideoResponse)
async def create_video_360(
    background_tasks: BackgroundTasks,
    request: Request,
    prompt: str = Form(None),
    model: Optional[str] = Form(None),
    size: Optional[str] = Form(None),
    seconds: Optional[str] = Form(None),
    quality: Optional[str] = Form(None),
    image: UploadFile = File(None),
    image_url: Optional[str] = Form(None),
):
    """创建360度旋转视频(POST /v1/videos/360)

    360-Degree Character Rotation - 基于输入图像生成360度旋转视频

    支持两种请求格式:
    1. multipart/form-data: 本地文件或URL链接
       - image: 本地图像文件
       - image_url: 图像URL链接
    2. application/json: URL链接或Base64
       - image: 图片URL或base64

    Args:
        prompt: 文本提示词(必填)
        model: 模型名称
        size: 分辨率(推荐1280x720)
        seconds: 时长(推荐5秒确保完整360度旋转)
        quality: 质量
        image: 本地图像文件/URL/base64(必填)

    Returns:
        视频任务响应
    """

    # 检测请求类型并提取参数
    content_type = request.headers.get("content-type", "")
    
    # JSON格式请求
    if "application/json" in content_type or prompt is None:
        try:
            body = await request.json()
            prompt = body.get("prompt")
            model = body.get("model")
            size = body.get("size")
            seconds = body.get("seconds")
            quality = body.get("quality")
            json_image = body.get("image")
            
            if not json_image:
                raise HTTPException(
                    status_code=400,
                    detail="Image is required for 360-degree video generation."
                )
            
            # 处理JSON图片数据 - URL或Base64
            image_data = None
            
            try:
                if json_image.startswith('data:image') or len(json_image) > 200:
                    import base64
                    if ',' in json_image:
                        base64_data = json_image.split(',', 1)[1]
                    else:
                        base64_data = json_image
                    image_data = bytearray(base64.b64decode(base64_data))
                else:
                    import requests as req_lib
                    response = req_lib.get(json_image, timeout=30)
                    response.raise_for_status()
                    image_data = bytearray(response.content)
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to process image: {str(e)}"
                )
            
            # 设置默认值 - 360度推荐5秒
            model = model or API_CONFIG.default_model
            size = size or "1280x720"
            seconds = seconds or "5"
            
            if not validate_openai_params(model=model, size=size, seconds=seconds):
                raise HTTPException(
                    status_code=400,
                    detail="Invalid parameter value"
                )
            
            # 创建任务
            task_manager = get_task_manager()
            
            task = await task_manager.create_task(
                prompt=prompt,
                model=model,
                size=size,
                seconds=seconds,
                quality=quality or API_CONFIG.default_quality,
                video_type="360",
            )
            
            background_tasks.add_task(
                run_generation_task_360,
                task_id=task.id,
                prompt=prompt,
                image_data=image_data,
                size=size,
                seconds=seconds,
            )
            
            return task_to_response(task)
            
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid JSON request: {str(e)}"
            )
    
    # Form表单格式请求 (原有逻辑)
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

    task = await task_manager.create_task(
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
      task = await task_manager.get_task(video_id)
      
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
      task = await task_manager.get_task(video_id)
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
      success = await task_manager.delete_task(video_id)
      
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
    task = await task_manager.get_task(video_id)

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
