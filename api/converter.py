# -*- coding: utf-8 -*-
"""
参数转换模块 - OpenAI API参数转换为AnisoraV3参数
"""
import logging
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass, field
from PIL import Image
import io
import os
import tempfile

from .config import GENERATION_CONFIG as GEN_CFG

logger = logging.getLogger(__name__)

# 默认提示词后缀（质量评分）
DEFAULT_PROMPT_SUFFIX = "aesthetic score: 5.5. motion score: 3.0. There is no text in the video."

# 图片权重配置
IMAGE_WEIGHT_CONFIGS = {
    1: "&&1",      # 单图主引导
    2: "&&1,1",   # 双图首尾帧
    3: "&&0.3,0.7,1",  # 三张图
}


def enhance_prompt(prompt: str) -> str:
    """增强提示词，追加质量评分参数
    
    Args:
        prompt: 原始提示词
        
    Returns:
        增强后的提示词
    """
    # 检查是否已包含质量评分后缀
    if "aesthetic score:" in prompt.lower() or "motion score:" in prompt.lower():
        return prompt
    
    # 追加默认后缀
    return f"{prompt}. {DEFAULT_PROMPT_SUFFIX}"


def get_image_weight_config(image_count: int) -> str:
    """获取图片权重配置
    
    Args:
        image_count: 图片数量
        
    Returns:
        权重配置字符串，如 "&&1" 或 "&&1,1"
    """
    if image_count in IMAGE_WEIGHT_CONFIGS:
        return IMAGE_WEIGHT_CONFIGS[image_count]
    
    # 默认单图配置
    return IMAGE_WEIGHT_CONFIGS[1]


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
    return second_int * 16 + 1


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
        
        # 设置提示词 - 增强提示词追加质量评分参数
        params.prompt = enhance_prompt(request.prompt)
        
        # base_seed 使用 dataclass 默认值 4096
        
        # 使用默认采样参数(从GENERATION_CONFIG读取)
        params.sample_step = GEN_CFG.sample_step
        
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
    
    def convert_with_images(
        self, 
        request: OpenAICreateVideoRequest, 
        images_data_list: List[bytes], 
        weight_config: str = "&&1",
        work_dir: str = None
    ) -> AnisoraParams:
        """将OpenAI请求(带多张图像)转换为AnisoraV3参数
        
        Args:
            request: OpenAI请求对象
            images_data_list: 图片数据列表
            weight_config: 图片权重配置，如"&&1"或"&&1,1"
            work_dir: 工作目录
            
        Returns:
            AnisoraParams参数对象
        """
        # 先调用基础转换方法获取基本参数对象
        params = self.convert(request)
        
        # 保存图像到指定目录或临时文件
        if work_dir:
            image_dir = work_dir
            os.makedirs(image_dir, exist_ok=True)
            temp_dir = None
        else:
            temp_dir = tempfile.mkdtemp(prefix="anisora_input_")
            image_dir = temp_dir
        
        # 根据图片数量设置文件名
        image_names = []
        
        for idx, image_data in enumerate(images_data_list):
            # 先设置图片路径
            image_name = f"input_image_{idx}.jpg"
            image_path = os.path.join(image_dir, image_name)
            
            # 根据目标尺寸调整图像(保持宽高比)
            target_width, target_height = map(int, params.size.replace('*', 'x').split('x'))
            
            img = Image.open(io.BytesIO(image_data))
            
            img_width, img_height = img.size
            aspect_ratio_target = target_width / target_height  
            aspect_ratio_image = img_width / img_height
            
            # 处理图像并保存
            if aspect_ratio_image > aspect_ratio_target:
                new_height = target_height
                new_width = int(new_height * aspect_ratio_image)
                img_resized = img.resize((new_width, new_height), Image.LANCZOS)
                crop_left = (new_width - target_width) // 2
                img_final = img_resized.crop((crop_left, 0, crop_left + target_width, target_height))
            elif aspect_ratio_image < aspect_ratio_target:
                new_width = target_width
                new_height = int(new_width / aspect_ratio_image)
                img_resized = img.resize((new_width, new_height), Image.LANCZOS)
                crop_top = (new_height - target_height) // 2
                img_final = img_resized.crop((0, crop_top, target_width, crop_top + target_height))
            else:
                img_final = img.resize((target_width, target_height), Image.LANCZOS)
            
            # 保存图片
            img_final.save(image_path, quality=95)
            image_names.append(image_name)
        
        # 设置图像路径（多图用逗号分隔）和权重配置
        # 格式: image1.jpg,image2.jpg@@prompt&&weight_config
        images_str = ",".join(image_names)
        
        # 修改prompt添加权重配置：prompt@@image1.jpg,image2.jpg&&weight_config
        enhanced_prompt = f"{params.prompt}@@{images_str}{weight_config}"
        
        params.prompt = enhanced_prompt
        # 设置第一个图片的完整路径，让generator识别为I2V任务
        params.image = os.path.join(image_dir, image_names[0])
        
        if temp_dir:
            params.temp_dirs.append(temp_dir)
        
        logger.info(f"Created multi-image prompt: {enhanced_prompt}, first image: {params.image}")
        
        return params
