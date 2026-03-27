# -*- coding: utf-8 -*-
"""
参数转换模块 - OpenAI API参数转换为AnisoraV3参数
"""
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from PIL import Image
import io
import os
import tempfile

from .config import GENERATION_CONFIG as GEN_CFG


@dataclass
class AnisoraParams:
    """AnisoraV3生成参数 - 与官方generate-pi-i2v-any.py参数一致"""
    task: str = "i2v-14B"
    size: str = "480*832"
    frame_num: int = 81
    ckpt_dir: str = "/root/xinglin-data/chat/Index-anisora/V3.1"
    prompt: str = ""
    image: Optional[str] = None
    base_seed: int = 4096
    sample_step: int = 8
    sample_shift: float = 5.0
    sample_guide_scale: float = 1.0
    sample_solver: str = "unipc"
    t5_fsdp: bool = True
    dit_fsdp: bool = True
    t5_cpu: bool = False
    offload_model: bool = True
    ulysses_size: int = 8
    ring_size: int = 1
    temp_dirs: List[str] = field(default_factory=list)


@dataclass  
class OpenAICreateVideoRequest:
    """OpenAI创建视频请求参数"""
    prompt: str
    model: Optional[str] = None
    size: Optional[str] = None
    seconds: Optional[str] = None
    quality: Optional[str] = None


def get_frame_number_for_seconds(seconds_str: str) -> int:
    """根据秒数计算帧数"""
    second_int = int(seconds_str) if seconds_str else 4
    return second_int * 8 + 1


class ParameterConverter:
    """参数转换器 - 将OpenAI参数转换为AnisoraV3参数"""
    
    def __init__(self, checkpoint_dir: str = "/root/xinglin-data/chat/Index-anisora/V3.1"):
        self.checkpoint_dir = checkpoint_dir
    
    def convert(self, request: OpenAICreateVideoRequest) -> AnisoraParams:
        """将OpenAI请求转换为AnisoraV3参数(T2V模式)"""
        params = AnisoraParams()
        
        params.task = "t2v-14B"  # T2V任务类型
        params.ckpt_dir = self.checkpoint_dir
        
        # 将OpenAI尺寸映射到Wan格式 (如720x1280 -> 480*832)
        size_map_to_wan_format = {
            "720x1280": "480*832",
            "1280x720": "832*480",
            "1024x1792": "704*1216", 
            "1792x1024": "1216*704",
        }
        
        input_size = request.size or "720x1280"
        
        if input_size in size_map_to_wan_format:
            params.size = size_map_to_wan_format[input_size]
        else:
            params.size = input_size
        
        # 获取秒数并计算帧数
        seconds_str = request.seconds or "4"
        params.frame_num = get_frame_number_for_seconds(seconds_str)
        
        # 设置提示词和种子
        params.prompt = request.prompt
        
        import time
        params.base_seed = int(time.time()) % (2**31)
        
        # 使用默认采样参数(从GENERATION_CONFIG读取)
        params.sample_step = GEN_CFG.sample_steps
        
        # 根据分辨率动态设置sample_shift (与官方逻辑一致)
        if params.size == "480*832" or params.size == "832*480":
            params.sample_shift = 3.0
        else:
            params.sample_shift = GEN_CFG.sample_shift
        
        params.sample_guide_scale = GEN_CFG.sample_guide_scale
        params.sample_solver = GEN_CFG.sample_solver
        
        # 设置分布式参数(8卡)
        params.t5_fsdp = True
        params.dit_fsdp = True
        params.ulysses_size = 8
        
        return params
    
    def convert_with_image(self, request: OpenAICreateVideoRequest, image_data: bytes, work_dir: str = None) -> AnisoraParams:
        """将OpenAI请求(带图像)转换为AnisoraV3参数"""
        # 先调用基础转换方法获取基本参数对象
        params = self.convert(request)
        
        # 保存图像到指定目录或临时文件
        if work_dir:
            # 使用传入的工作目录
            image_dir = work_dir
            os.makedirs(image_dir, exist_ok=True)
            temp_dir = None
        else:
            # 兼容旧逻辑：创建临时目录
            temp_dir = tempfile.mkdtemp(prefix="anisora_input_")
            image_dir = temp_dir
        
        image_path = os.path.join(image_dir, "input_image.jpg")
        
        # 从bytes创建PIL Image并保存
        img = Image.open(io.BytesIO(image_data))
        
        # 根据目标尺寸调整图像(保持宽高比)
        target_width, target_height = map(int, params.size.replace('*', 'x').split('x'))
        
        img_width, img_height = img.size
        aspect_ratio_target = target_width / target_height  
        aspect_ratio_image = img_width / img_height
        
        if aspect_ratio_image > aspect_ratio_target:
            new_height = target_height
            new_width = int(new_height * aspect_ratio_image)
            img_resized = img.resize((new_width, new_height), Image.LANCZOS)
            crop_left = (new_width - target_width) // 2
            img_final = img_resized.crop((crop_left, 0, crop_left + target_width, target_height))
            img_final.save(image_path, quality=95)
        elif aspect_ratio_image < aspect_ratio_target:
            new_width = target_width
            new_height = int(new_width / aspect_ratio_image)
            img_resized = img.resize((new_width, new_height), Image.LANCZOS)
            crop_top = (new_height - target_height) // 2
            img_final = img_resized.crop((0, crop_top, target_width, crop_top + target_height))
            img_final.save(image_path, quality=95)
        else:
            img_resized = img.resize((target_width, target_height), Image.LANCZOS)
            img_resized.save(image_path, quality=95)
        
        # 设置图像路径和临时目录用于后续清理
        params.image = image_path
        if temp_dir:
            params.temp_dirs.append(temp_dir)
        
        return params
