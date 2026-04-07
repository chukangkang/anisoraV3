# -*- coding: utf-8 -*-
"""
任务管理模块 - 异步任务状态管理
"""
import os
import time
import uuid
import logging
import asyncio
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum

from .config import VideoStatus


logger = logging.getLogger(__name__)


@dataclass
class VideoTask:
    """视频生成任务"""
    id: str                          # 任务ID
    prompt: str                     # 提示词
    model: str = "sora-2"          # 模型名称
    size: str = "720x1280"        # 分辨率
    seconds: str = "4"            # 时长(秒)
    quality: str = "standard"     # 质量
    status: str = "queued"        # 状态
    progress: int = 0             # 进度(0-100)
    created_at: int = 0          # 创建时间戳
    completed_at: Optional[int] = None  # 完成时间戳
    video_path: Optional[str] = None  # 视频文件路径
    error: Optional[str] = None   # 错误信息
    video_type: Optional[str] = None  # 视频类型(normal/360)
    
    
class TaskManager:
    """任务管理器 - 管理异步视频生成任务"""
    
    def __init__(self):
        """初始化任务管理器"""
        self._tasks: Dict[str, VideoTask] = {}
        self._lock = asyncio.Lock()
        
    async def create_task(
        self,
        prompt: str,
        model: str = "sora-2",
        size: str = "720x1280",
        seconds: str = "4",
        quality: str = "standard",
        video_type: Optional[str] = None,
    ) -> VideoTask:
        """创建新任务
        
        Args:
            prompt: 提示词
            model: 模型名称
            size: 分辨率
            seconds: 时长(秒)
            quality: 质量
            video_type: 视频类型(normal/360)
            
        Returns:
            创建的任务对象
        """
        task_id = f"video_{uuid.uuid4().hex[:12]}"
        
        task = VideoTask(
            id=task_id,
            prompt=prompt,
            model=model,
            size=size,
            seconds=seconds,
            quality=quality,
            status=VideoStatus.QUEUED.value,
            progress=0,
            created_at=int(time.time()),
            video_type=video_type,
        )
        
        async with self._lock:
            self._tasks[task_id] = task
            
        logger.info(f"Created task: {task_id}")
        
        return task
    
    async def get_task(self, task_id: str) -> Optional[VideoTask]:
        """获取任务
        
        Args:
            task_id: 任务ID
            
        Returns:
            任务对象，不存在返回None
        """
        async with self._lock:
            return self._tasks.get(task_id)
    
    async def update_task(
        self,
        task_id: str,
        status: Optional[str] = None,
        progress: Optional[int] = None,
        video_path: Optional[str] = None,
        error: Optional[str] = None,
    ) -> bool:
        """更新任务状态
        
        Args:
            task_id: 任务ID  
            status: 新状态
            progress: 新进度(0-100)
            video_path: 视频文件路径
            error: 错误信息
            
        Returns:
            是否更新成功
        """
        async with self._lock:
            if task_id not in self._tasks:
                return False
            
            task = self._tasks[task_id]
            
            if status:
                task.status = status
                
                if status == VideoStatus.COMPLETED.value or status == VideoStatus.FAILED.value:
                    task.completed_at = int(time.time())
                    
            if progress is not None:
                task.progress = min(100, max(0, progress))
                
                if progress > 0 and progress < 100:
                    task.status = VideoStatus.PROCESSING.value
                    
            if video_path:
                task.video_path = video_path
                
            if error:
                task.error = error
            
            return True
    
    async def list_tasks(self, limit: int = 100) -> List[VideoTask]:
         """列出所有任务 
         
         Args:
             limit: 返回数量限制  
             
         Returns:
             任务列表(按创建时间倒序)  
         """
         async with self._lock:
             tasks = sorted(
                 self._tasks.values(), 
                 key=lambda t: t.created_at, 
                 reverse=True 
             )
             return tasks[:limit]
    
    async def delete_task(self, task_id: str) -> bool:
         """删除任务  
         
         Args:
             task_id: 任务ID  
             
         Returns:
             是否删除成功  
         """
         async with self._lock:
             if task_id in self._tasks:
                 del self._tasks[task_id]
                 return True 
             return False


# 全局任务管理器实例  
_task_manager_instance: Optional[TaskManager] = None


def get_task_manager() -> TaskManager:
     """获取全局任务管理器实例(单例模式)  
     
     Returns:
          TaskManager实例  
     """
     global _task_manager_instance 
     
     if _task_manager_instance is None:
          _task_manager_instance = TaskManager()
     
     return _task_manager_instance


def reset_task_manager():
      """重置全局任务管理器实例"""
      global _task_manager_instance
      
      _task_manager_instance = None
