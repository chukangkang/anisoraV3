# -*- coding: utf-8 -*-
"""
AnisoraV3 API - OpenAI兼容的视频生成API
"""
from .config import (
    API_CONFIG,
    MODEL_CONFIG,
    DISTRIBUTED_CONFIG,
    GENERATION_CONFIG,
    VideoModel,
    VideoSize,
    VideoSeconds,
    VideoQuality,
    VideoStatus,
)
from .converter import (
    AnisoraParams,
    OpenAICreateVideoRequest,
    ParameterConverter,
)
from .model_loader import (
    DistributedModelLoader,
    SingleModelLoader,
    create_model_loader,
)
from .generator import (
    VideoGenerator,
    get_generator,
    reset_generator,
)
from .task_manager import (
    VideoTask,
    TaskManager,
    get_task_manager,
    reset_task_manager,
)

__version__ = "1.0.0"
__all__ = [
    # Config
    "API_CONFIG",
    "MODEL_CONFIG", 
    "DISTRIBUTED_CONFIG",
    "GENERATION_CONFIG",
    "VideoModel",
    "VideoSize",
    "VideoSeconds", 
    "VideoQuality",
    "VideoStatus",
    
    # Converter
    "AnisoraParams",
    "OpenAICreateVideoRequest", 
    "ParameterConverter",
    
    # Model Loader
    "DistributedModelLoader",
    "SingleModelLoader", 
    "create_model_loader",
    
    # Generator
    "VideoGenerator",
    "get_generator", 
    "reset_generator",
    
    # Task Manager
    "VideoTask",
    "TaskManager", 
    "get_task_manager",
]
