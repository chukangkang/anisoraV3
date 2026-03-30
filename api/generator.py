# -*- coding: utf-8 -*-
"""
视频生成器 - 核心生成逻辑
通过torchrun调用generate-pi-i2v-any.py脚本执行推理
"""
import os
import sys
import logging
import time
import uuid
import subprocess
import tempfile
import shutil
from typing import Optional, Dict, Any, List
from pathlib import Path

import torch
from PIL import Image

from .converter import AnisoraParams, ParameterConverter


logger = logging.getLogger(__name__)


class VideoGenerator:
    """视频生成器 - 通过torchrun调用generate-pi-i2v-any.py脚本"""
    
    def __init__(
        self,
        checkpoint_dir: str = "Wan2.1-I2V-14B-480P",
        output_dir: str = "output_videos_any",
        distributed: bool = True,
        world_size: int = 8,
    ):
        """初始化视频生成器
        
        Args:
            checkpoint_dir: 模型检查点目录
            output_dir: 输出目录
            distributed: 是否使用分布式(8卡)
            world_size: GPU数量
        """
        self.checkpoint_dir = checkpoint_dir
        self.output_dir = output_dir
        self.distributed = distributed
        self.world_size = world_size
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 参数转换器
        self.converter = ParameterConverter(checkpoint_dir=checkpoint_dir)
    
    def generate(
        self,
        prompt: str,
        image_path: Optional[str] = None,
        size: str = "720x1280",
        seconds: str = "4",
        seed: int = -1,
        task_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """生成视频 - 通过torchrun调用脚本执行
        
        Args:
            prompt: 文本提示词
            image_path: 输入图像路径(Optional)
            size: 分辨率 
            seconds: 时长(秒)
            seed: 随机种子
            task_id: 任务ID，用于命名输出文件
            
        Returns:
            生成结果字典 {
                "success": bool,
                "video_path": str,  # 视频文件路径
                "error": str,       # 错误信息(如果失败)
                "metadata": dict   # 元数据
            }
        """
        # 如果没有提供task_id，生成一个唯一ID
        if not task_id:
            task_id = uuid.uuid4().hex[:8]
        # 构建请求对象
        from .converter import OpenAICreateVideoRequest

        request = OpenAICreateVideoRequest(
            prompt=prompt,
            size=size,
            seconds=seconds,
        )
        
        # 转换参数
        if image_path:
            # 带图像的转换(从文件读取)
            with open(image_path, 'rb') as f:
                image_data = f.read()
            params = self.converter.convert_with_image(request, image_data)
        else:
            params = self.converter.convert(request)
        
        # 如果提供了seed，覆盖默认seed
        if seed >= 0:
            params.base_seed = seed
        
        try:
            # 执行生成 - 通过torchrun调用脚本
            logger.info(f"Starting video generation via torchrun: prompt={prompt[:50]}..., size={size}, seconds={seconds}")
            
            start_time = time.time()
            
            video_path = self._generate_via_torchrun(params, prompt, image_path, task_id)
            
            elapsed_time = time.time() - start_time
            
            if video_path and os.path.exists(video_path):
                return {
                    "success": True,
                    "video_path": video_path,
                    "error": None,
                    "metadata": {
                        "prompt": prompt,
                        "size": size,
                        "seconds": seconds,
                        "frame_num": params.frame_num,
                        "seed": params.base_seed,
                        "elapsed_time": elapsed_time,
                    },
                }
            else:
                return {
                    "success": False,
                    "video_path": None,
                    "error": f"Video generation failed or output not found",
                    "metadata": {},
                }
            
        except Exception as e:
            logger.error(f"Failed to generate video: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                "success": False,
                "video_path": None,
                "error": str(e),
                "metadata": {},
            }

    def generate_360(
        self,
        prompt: str,
        image_path: Optional[str] = None,
        size: str = "1280x720",
        seconds: str = "5",
        seed: int = -1,
        task_id: Optional[str] = None,
        frame_num: int = 81,
    ) -> Dict[str, Any]:
        """生成360度旋转视频 - 通过torchrun调用脚本执行

        Args:
            prompt: 文本提示词(格式: image_path@@prompt&&image_position)
            image_path: 输入图像路径(Optional)
            size: 分辨率 (推荐1280x720)
            seconds: 时长(推荐5秒确保完整360度旋转)
            seed: 随机种子
            task_id: 任务ID，用于命名输出文件
            frame_num: 帧数(默认81，对应5秒@16fps)

        Returns:
            生成结果字典 {
                "success": bool,
                "video_path": str,  # 视频文件路径
                "error": str,       # 错误信息(如果失败)
                "metadata": dict   # 元数据
            }
        """
        # 如果没有提供task_id，生成一个唯一ID
        if not task_id:
            task_id = uuid.uuid4().hex[:8]

        # 构建请求对象
        from .converter import OpenAICreateVideoRequest

        request = OpenAICreateVideoRequest(
            prompt=prompt,
            size=size,
            seconds=seconds,
        )

        # 转换参数 - 使用默认配置
        params = self.converter.convert(request)

        # 如果提供了seed，覆盖默认seed
        if seed >= 0:
            params.base_seed = seed

        # 覆盖frame_num为81 (360度推荐)
        params.frame_num = frame_num

        try:
            # 执行360度生成 - 通过torchrun调用脚本
            logger.info(f"Starting 360-degree video generation via torchrun: prompt={prompt[:50]}..., size={size}, seconds={seconds}, frame_num={frame_num}")

            start_time = time.time()

            video_path = self._generate_via_torchrun_360(params, prompt, image_path, task_id)

            elapsed_time = time.time() - start_time

            if video_path and os.path.exists(video_path):
                return {
                    "success": True,
                    "video_path": video_path,
                    "error": None,
                    "metadata": {
                        "prompt": prompt,
                        "size": size,
                        "seconds": seconds,
                        "frame_num": frame_num,
                        "seed": params.base_seed,
                        "elapsed_time": elapsed_time,
                        "video_type": "360",
                    },
                }
            else:
                return {
                    "success": False,
                    "video_path": None,
                    "error": f"360 video generation failed or output not found",
                    "metadata": {},
                }

        except Exception as e:
            logger.error(f"Failed to generate 360 video: {e}")
            import traceback
            traceback.print_exc()

            return {
                "success": False,
                "video_path": None,
                "error": str(e),
                "metadata": {},
            }

    def _generate_via_torchrun_360(self, params: AnisoraParams, prompt: str, image_path: Optional[str] = None, task_id: str = None) -> Optional[str]:
        """通过torchrun调用generate-pi-i2v-any.py脚本执行360度旋转视频生成

        Args:
            params: 生成参数
            prompt: 文本提示词
            image_path: 输入图像路径(优先使用)
            task_id: 任务ID

        Returns:
            生成的视频文件路径，失败返回None
        """

        # 创建临时工作目录 - 使用统一输入目录，以任务ID命名
        project_root = "/root/anisoraV3"
        input_base_dir = os.path.join(project_root, "anisora_input")

        # 创建统一输入目录
        os.makedirs(input_base_dir, exist_ok=True)

        # 创建任务专属目录 - 使用任务ID命名 (video_xxx格式)
        work_dir = os.path.join(input_base_dir, task_id)
        os.makedirs(work_dir, exist_ok=True)

        logger.info(f"360 Work directory: {work_dir}")

        # 准备prompt文件 - 格式: prompt@@image_path (相对于work_dir的路径)
        prompt_file_name = "prompt.txt"

        # 判断是I2V还是T2V任务 - 优先使用image_path参数
        actual_image_path = None

        if image_path and os.path.exists(image_path):
            # I2V模式 - 使用传入的图像路径
            actual_image_path = image_path
            
            # I2V模式 - 复制图像并写入带图像的prompt文件
            input_image_name = os.path.basename(image_path)
            input_image_path_in_workdir = os.path.join(work_dir, input_image_name)
            
            # 如果源文件和目标文件相同，跳过复制
            if os.path.abspath(image_path) != os.path.abspath(input_image_path_in_workdir):
                shutil.copy(image_path, input_image_path_in_workdir)
                logger.info(f"Copied 360 image to work directory: {input_image_path_in_workdir}")
            else:
                logger.info(f"Image already in work directory: {input_image_path_in_workdir}")
            
            # 写入prompt文件 - 格式: prompt@@image_path&&time_position (与官方脚本一致)
            with open(os.path.join(work_dir, prompt_file_name), 'w', encoding='utf-8') as f:
                f.write(f"{prompt}@@{input_image_name}&&0.0")
            
            logger.info(f"Prepared 360 I2V prompt file in {work_dir}: {prompt}@@{input_image_name}&&0.0")
            
            is_i2v = True
            task_type = "i2v-14B"
        elif "@@" in prompt:
            # 从prompt中提取图像路径 (兼容旧格式)
            image_part = prompt.split("@@")[0]
            if os.path.exists(image_part):
                actual_image_path = image_part
                # 获取纯文本prompt部分
                prompt_text = prompt.split("@@")[1].split("&&")[0] if "&&" in prompt else prompt.split("@@")[1]
                
                # I2V模式 - 复制图像并写入带图像的prompt文件
                input_image_name = os.path.basename(image_part)
                input_image_path_in_workdir = os.path.join(work_dir, input_image_name)
                
                shutil.copy(image_part, input_image_path_in_workdir)
                logger.info(f"Copied 360 image to work directory: {input_image_path_in_workdir}")
                
                # 写入prompt文件 - 格式: prompt@@image_path&&time_position (与官方脚本一致)
                with open(os.path.join(work_dir, prompt_file_name), 'w', encoding='utf-8') as f:
                    f.write(f"{prompt_text}@@{input_image_name}&&0.0")
                
                logger.info(f"Prepared 360 I2V prompt file in {work_dir}: {prompt_text}@@{input_image_name}&&0.0")
                
                is_i2v = True
                task_type = "i2v-14B"
            else:
                # T2V模式 - 只写入纯文本prompt文件
                with open(os.path.join(work_dir, prompt_file_name), 'w', encoding='utf-8') as f:
                    f.write(prompt)
                
                logger.info(f"Prepared 360 T2V prompt file in {work_dir}: {prompt}")
                
                is_i2v = False
                task_type = "t2v-14B"
        else:
            # 没有图像路径，纯T2V模式
            with open(os.path.join(work_dir, prompt_file_name), 'w', encoding='utf-8') as f:
                f.write(prompt)
            
            logger.info(f"Prepared 360 T2V prompt file in {work_dir}: {prompt}")
            
            is_i2v = False
            task_type = "t2v-14B"

        # 输出目录 - 使用工作目录下的output子目录 (与官方脚本一致)
        output_dir_name = "output"

        # 构建torchrun命令 - 参考官方启动参数，使用绝对路径调用脚本
        cmd = [
            "torchrun",
            f"--nproc_per_node={self.world_size}",
            "--master_port=43210",
            os.path.join(project_root, "generate-pi-i2v-any.py"),
            "--task", task_type,
            "--size", params.size.replace('x', '*'),  # 480*832格式
            "--ckpt_dir", self.checkpoint_dir,
        ]

        # 根据任务类型添加不同参数
        if is_i2v:
            cmd.extend([
                "--image", output_dir_name,  # 使用相对路径，脚本会在work_dir下创建此目录
                "--prompt", prompt_file_name,
                "--dit_fsdp",
                "--t5_fsdp",
                "--offload_model", "true" if params.offload_model else "false",
                "--ulysses_size", str(self.world_size),
                "--base_seed", str(params.base_seed),
                "--frame_num", str(params.frame_num),
                "--sample_step", str(params.sample_step),
                "--sample_shift", str(params.sample_shift),
                "--sample_guide_scale", str(params.sample_guide_scale),
            ])
        else:
            cmd.extend([
                "--prompt", prompt_file_name,
                "--dit_fsdp",
                "--t5_fsdp",
                "--offload_model", "true" if params.offload_model else "false",
                "--ulysses_size", str(self.world_size),
                "--base_seed", str(params.base_seed),
                "--frame_num", str(params.frame_num),
                "--sample_step", str(params.sample_step),
                "--sample_shift", str(params.sample_shift),
                "--sample_guide_scale", str(params.sample_guide_scale),
            ])

        logger.info(f"Running 360 torchrun command in {work_dir}: {' '.join(cmd)}")

        # 执行命令 - 在工作目录下执行确保能找到相对路径的文件和模块
        try:
            result = subprocess.run(
                cmd,
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=3600,  # 1小时超时
                env={
                    **os.environ, 
                    "HF_ENDPOINT": "https://hf-mirror.com",
                    "PYTHONPATH": project_root + ":" + os.environ.get("PYTHONPATH", ""),
                }
            )

            logger.info(f"360 Torchrun stdout:\n{result.stdout[-5000:] if len(result.stdout) > 5000 else result.stdout}")  # 只打印最后5000字符

            if result.returncode != 0:
                logger.error(f"360 Torchrun stderr:\n{result.stderr[-5000:] if len(result.stderr) > 5000 else result.stderr}")
                return None

            # 查找生成的视频文件 - 在output目录下查找最新的mp4文件 (与官方脚本一致)
            output_full_dir = os.path.join(work_dir, output_dir_name)

            if not os.path.exists(output_full_dir):
                logger.error(f"360 Output directory not found: {output_full_dir}")
                return None

            video_file = list(Path(output_full_dir).glob("*.mp4"))

            if not video_file:
                logger.error(f"No video file found in {output_full_dir}")
                return None

            # 获取最新的视频文件 (按修改时间排序，取最新的)
            latest_video = max(video_file, key=lambda p: p.stat().st_mtime)

            logger.info(f"Found generated 360 video: {latest_video}")

            # 复制到目标输出目录使用任务ID命名 (task_id已经是video_xxx格式)
            final_filename = f"{task_id}.mp4"
            final_path = os.path.join(self.output_dir, final_filename)

            shutil.copy(latest_video, final_path)

            logger.info(f"360 Video copied to final path: {final_path}")

            return final_path

        except subprocess.TimeoutExpired:
            logger.error("360 Torchrun command timed out")
            return None

        except Exception as e:
            logger.error(f"Error during 360 torchrun execution: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _generate_via_torchrun(self, params: AnisoraParams, prompt: str, image_path: Optional[str], task_id: str) -> Optional[str]:
        """通过torchrun调用generate-pi-i2v-any.py脚本执行I2V生成
        
        Args:
            params: 生成参数
            prompt: 文本提示词
            image_path: 输入图像路径
            task_id: 任务ID
            
        Returns:
            生成的视频文件路径，失败返回None
        """
        
        # 创建临时工作目录 - 使用统一输入目录，以任务ID命名
        project_root = "/root/anisoraV3"
        input_base_dir = os.path.join(project_root, "anisora_input")
        
        # 创建统一输入目录
        os.makedirs(input_base_dir, exist_ok=True)
        
        # 创建任务专属目录 - 使用任务ID命名 (video_xxx格式)
        work_dir = os.path.join(input_base_dir, task_id)
        os.makedirs(work_dir, exist_ok=True)
        
        logger.info(f"Work directory: {work_dir}")
        
        # 准备prompt文件 - 格式: prompt@@image_path (相对于work_dir的路径)
        prompt_file_name = "prompt.txt"
        
        # 判断是I2V还是T2V任务
        is_i2v = image_path and os.path.exists(image_path)
        
        if is_i2v:
            # I2V模式 - 复制图像并写入带图像的prompt文件
            input_image_name = "input_image.jpg"
            input_image_path_in_workdir = os.path.join(work_dir, input_image_name)
            
            # 如果图像已经在工作目录中（由converter保存），则跳过复制
            if os.path.abspath(image_path) != os.path.abspath(input_image_path_in_workdir):
                shutil.copy(image_path, input_image_path_in_workdir)
                logger.info(f"Copied image to work directory: {input_image_path_in_workdir}")
            else:
                logger.info(f"Image already in work directory: {input_image_path_in_workdir}")
            
            # 写入prompt文件 - 格式: prompt@@image_path&&time_position (与官方脚本一致)
            with open(os.path.join(work_dir, prompt_file_name), 'w', encoding='utf-8') as f:
                f.write(f"{prompt}@@{input_image_name}&&0.0")
            
            logger.info(f"Prepared I2V prompt file in {work_dir}: {prompt}@@{input_image_name}&&0.0")
            
            task_type = "i2v-14B"
        else:
            # T2V模式 - 只写入纯文本prompt文件
            with open(os.path.join(work_dir, prompt_file_name), 'w', encoding='utf-8') as f:
                f.write(prompt)
            
            logger.info(f"Prepared T2V prompt file in {work_dir}: {prompt}")
            
            task_type = "t2v-14B"
        
        # 输出目录 - 使用工作目录下的output子目录 (与官方脚本一致)
        output_dir_name = "output"
        
        # 构建torchrun命令 - 参考官方启动参数，使用绝对路径调用脚本
        cmd = [
            "torchrun",
            f"--nproc_per_node={self.world_size}",
            "--master_port=43210",
            os.path.join(project_root, "generate-pi-i2v-any.py"),
            "--task", task_type,
            "--size", params.size.replace('x', '*'),  # 480*832格式
            "--ckpt_dir", self.checkpoint_dir,
        ]
        
        # 根据任务类型添加不同参数
        if is_i2v:
            cmd.extend([
                "--image", output_dir_name,  # 使用相对路径，脚本会在work_dir下创建此目录
                "--prompt", prompt_file_name,
                "--dit_fsdp",
                "--t5_fsdp",
                "--offload_model", "true" if params.offload_model else "false",
                "--ulysses_size", str(self.world_size),
                "--base_seed", str(params.base_seed),
                "--frame_num", str(params.frame_num),
                "--sample_step", str(params.sample_step),
                "--sample_shift", str(params.sample_shift),
                "--sample_guide_scale", str(params.sample_guide_scale),
            ])
        else:
            cmd.extend([
                "--prompt", prompt_file_name,
                "--dit_fsdp",
                "--t5_fsdp",
                "--offload_model", "true" if params.offload_model else "false",
                "--ulysses_size", str(self.world_size),
                "--base_seed", str(params.base_seed),
                "--frame_num", str(params.frame_num),
                "--sample_step", str(params.sample_step),
                "--sample_shift", str(params.sample_shift),
                "--sample_guide_scale", str(params.sample_guide_scale),
            ])
        
        logger.info(f"Running torchrun command in {work_dir}: {' '.join(cmd)}")
        
        # 执行命令 - 在工作目录下执行确保能找到相对路径的文件和模块
        try:
            result = subprocess.run(
                cmd,
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=3600,  # 1小时超时
                env={
                    **os.environ, 
                    "HF_ENDPOINT": "https://hf-mirror.com",
                    "PYTHONPATH": project_root + ":" + os.environ.get("PYTHONPATH", ""),
                }
            )
            
            logger.info(f"Torchrun stdout:\n{result.stdout[-5000:] if len(result.stdout) > 5000 else result.stdout}")  # 只打印最后5000字符
            
            if result.returncode != 0:
                logger.error(f"Torchrun stderr:\n{result.stderr[-5000:] if len(result.stderr) > 5000 else result.stderr}")
                return None
            
            # 查找生成的视频文件 - 在output目录下查找最新的mp4文件 (与官方脚本一致)
            output_full_dir = os.path.join(work_dir, output_dir_name)
            
            if not os.path.exists(output_full_dir):
                logger.error(f"Output directory not found: {output_full_dir}")
                return None
            
            video_files = list(Path(output_full_dir).glob("*.mp4"))
            
            if not video_files:
                logger.error(f"No video file found in {output_full_dir}")
                return None
            
            # 获取最新的视频文件 (按修改时间排序，取最新的)
            latest_video = max(video_files, key=lambda p: p.stat().st_mtime)
            
            logger.info(f"Found generated video: {latest_video}")
            
            # 复制到目标输出目录使用任务ID命名 (task_id已经是video_xxx格式)
            final_filename = f"{task_id}.mp4"
            final_path = os.path.join(self.output_dir, final_filename)
            
            shutil.copy(latest_video, final_path)
            
            logger.info(f"Video copied to final path: {final_path}")
            
            return final_path
            
        except subprocess.TimeoutExpired:
            logger.error("Torchrun command timed out")
            return None
            
        except Exception as e:
            logger.error(f"Error during torchrun execution: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _cleanup_temp_files(self, params: AnisoraParams):
        """清理临时文件
        
        Args:
            params: 生成参数
        """
        for temp_dir in getattr(params, 'temp_dirs', []):
            try:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                    logger.info(f"Cleaned up temp directory: {temp_dir}")
            except Exception as e:
                logger.warning(f"Failed to cleanup temp directory {temp_dir}: {e}")

    def cleanup(self):
        """清理资源"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# 全局生成器实例(单例模式)  
_generator_instance: Optional[VideoGenerator] = None


def get_generator(
    checkpoint_dir: str = "/root/xinglin-data/chat/Index-anisora/V3.1",
    output_dir: str = "/root/anisoraV3/output_videos_any",
    distributed: bool = True,
    world_size: int = 8,
) -> VideoGenerator:
    """获取全局生成器实例(单例模式)
    
    Args:
        checkpoint_dir: 模型检查点目录 (默认使用配置中的路径)
        output_dir: 输出目录 (默认使用项目下的output目录)
        distributed: 是否使用分布式(8卡)
        world_size: GPU数量
        
    Returns:
        VideoGenerator实例
    """
    global _generator_instance
    
    if _generator_instance is None:
        _generator_instance = VideoGenerator(
            checkpoint_dir=checkpoint_dir,
            output_dir=output_dir,
            distributed=distributed,
            world_size=world_size,
        )
    
    return _generator_instance


def reset_generator():
    """重置全局生成器实例"""
    global _generator_instance
    
    if _generator_instance:
        _generator_instance.cleanup()
        _generator_instance = None

