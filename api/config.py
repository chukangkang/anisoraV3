# -*- coding: utf-8 -*-
"""
API配置模块 - 集中管理API相关配置
"""
from easydict import EasyDict
from typing import Optional, List
from enum import Enum


class VideoModel(str, Enum):
    """支持的视频生成模型"""
    SORA_2 = "sora-2"
    SORA_2_PRO = "sora-2-pro"


class VideoSize(str, Enum):
    """支持的视频分辨率"""
    VERTICAL_720P = "720x1280"   # 竖屏 720P
    HORIZONTAL_720P = "1280x720"  # 横屏 720P
    VERTICAL_1024P = "1024x1792"  # 竖屏 1024P
    HORIZONTAL_1024P = "1792x1024" # 横屏 1024P


class VideoSeconds(str, Enum):
    """支持的视频时长(秒)"""
    FOUR_SECONDS = "4"
    FIVE_SECONDS = "5"  # 360度推荐
    EIGHT_SECONDS = "8"
    TWELVE_SECONDS = "12"


class VideoQuality(str, Enum):
    """视频质量等级"""
    STANDARD = "standard"
    HIGH = "high"


class VideoStatus(str, Enum):
    """视频任务状态"""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# API配置
API_CONFIG = EasyDict()

# 服务配置
API_CONFIG.host = "0.0.0.0"
API_CONFIG.port = 8000
API_CONFIG.workers = 1  # FastAPI workers数量

# 模型配置
API_CONFIG.default_model = VideoModel.SORA_2.value
API_CONFIG.default_size = VideoSize.VERTICAL_720P.value
API_CONFIG.default_seconds = VideoSeconds.FOUR_SECONDS.value
API_CONFIG.default_quality = VideoQuality.STANDARD.value

# AnisoraV3模型路径配置
MODEL_CONFIG = EasyDict()
MODEL_CONFIG.checkpoint_dir = "/root/xinglin-data/chat/Index-anisora/V3.1"  # 模型检查点目录
MODEL_CONFIG.task = "i2v-14B"  # 默认任务类型

# 分布式配置
DISTRIBUTED_CONFIG = EasyDict()
DISTRIBUTED_CONFIG.world_size = 8  # 8卡分布式
DISTRIBUTED_CONFIG.backend = "nccl"
DISTRIBUTED_CONFIG.master_port = 29500

# 生成参数配置 - 与官方generate-pi-i2v-any.py启动参数一致
GENERATION_CONFIG = EasyDict()
GENERATION_CONFIG.sample_steps = 8   # 采样步数 (官方默认40，但可配置)
GENERATION_CONFIG.sample_shift = 5.0  # shift参数 (官方默认5.0，480*832时为3.0)
GENERATION_CONFIG.sample_guide_scale = 1.0  # guidance scale (官方默认1.0)
GENERATION_CONFIG.sample_solver = "unipc"  # 采样求解器

# 输出目录配置
OUTPUT_CONFIG = EasyDict()
OUTPUT_CONFIG.output_dir = "/root/anisoraV3/output_videos_any"  # 输出视频目录

# 根据视频秒数计算帧数: seconds * fps(8) + 1
def calculate_frame_num(seconds: int) -> int:
    """根据秒数计算帧数
    
    Args:
        seconds: 视频时长(秒)
    
    Returns:
        帧数 (4n+1格式)
    """
    return seconds * 8 + 1


# OpenAI尺寸到AnisoraV3尺寸的映射
SIZE_MAPPING = {
    VideoSize.VERTICAL_720P.value: "720x1280",
    VideoSize.HORIZONTAL_720P.value: "1280x720",
    VideoSize.VERTICAL_1024P.value: "1024x1792",
    VideoSize.HORIZONTAL_1024P.value: "1792x1024",
}


# 支持的OpenAI参数值验证
def validate_openai_params(
    model: Optional[str] = None,
    size: Optional[str] = None,
    seconds: Optional[str] = None,
    quality: Optional[str] = None
) -> bool:
    """验证OpenAI参数是否支持
    
    Args:
        model: 模型名称
        size: 分辨率
        seconds: 时长
        quality: 质量
    
    Returns:
        是否支持
    """
    supported_models = [m.value for m in VideoModel]
    supported_sizes = [s.value for s in VideoSize]
    supported_seconds = [s.value for s in VideoSeconds]
    
    if model and model not in supported_models:
        return False
    if size and size not in supported_sizes:
        return False  
    if seconds and seconds not in supported_seconds:
        return False
    
    return True


def get_frame_num_from_seconds(seconds_str: str) -> int:
    """从OpenAI seconds参数获取帧数
    
    Args:
        seconds_str: OpenAI秒数字符串 ("4", "8", "12")
    
    Returns:
        对应的帧数 (33, 65, 97)
    """
    seconds_map = {
        VideoSeconds.FOUR_SECONDS.value: 33,
        VideoSeconds.FIVE_SECONDS.value: 81,  # 360度推荐
        VideoSeconds.EIGHT_SECONDS.value: 65,
        VideoSeconds.TWELVE_SECONDS.value: 97,
    }
    return seconds_map.get(seconds_str, 33)
