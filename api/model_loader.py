# -*- coding: utf-8 -*-
"""
分布式模型加载器 - 支持8卡分布式加载AnisoraV3模型
"""
import os
import logging
import torch
import torch.distributed as dist
from typing import Optional, Dict, Any
from functools import partial

from wan.configs import WAN_CONFIGS, get_optimal_window_size
from wan.image2video_any import WanI2V_any


logger = logging.getLogger(__name__)


class DistributedModelLoader:
    """分布式模型加载器 - 支持8卡分布式加载"""

    def __init__(
        self,
        checkpoint_dir: str = "Wan2.1-I2V-14B-480P",
        task: str = "i2v-14B",
        world_size: int = 8,
        master_port: int = 29500,
    ):
        """初始化分布式模型加载器

        Args:
            checkpoint_dir: 模型检查点目录
            task: 任务类型 (i2v-14B, t2v-14B等)
            world_size: GPU数量(默认8卡)
            master_port: 主进程端口
        """
        self.checkpoint_dir = checkpoint_dir
        self.task = task
        self.world_size = world_size
        self.master_port = master_port

        self.model = None
        self.is_initialized = False
        self.rank = 0
        
    def _init_distributed(self, local_rank: int = 0) -> bool:
        """初始化分布式环境
        
        Args:
            local_rank: 本地GPU编号
            
        Returns:
            是否成功初始化
        """
        # 检查是否已经初始化
        if dist.is_initialized():
            self.rank = dist.get_rank()
            logger.info(f"Distributed already initialized, rank={self.rank}")
            return True
        
        # 设置当前GPU
        torch.cuda.set_device(local_rank)
        
        # 初始化分布式进程组
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = str(self.master_port)
        
        try:
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                rank=local_rank,
                world_size=self.world_size
            )
            self.rank = local_rank
            logger.info(f"Initialized distributed: rank={local_rank}, world_size={self.world_size}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize distributed: {e}")
            return False
    
    def load_model(
        self,
        size: str = "720x1280",
        t5_fsdp: bool = True,
        dit_fsdp: bool = True,
        t5_cpu: bool = False,
    ) -> Optional[WanI2V_any]:
        """加载模型(分布式)
        
        Args:
            size: 视频分辨率
            t5_fsdp: 是否对T5使用FSDP
            dit_fsdp: 是否对DiT使用FSDP  
            t5_cpu: 是否将T5放在CPU上
            
        Returns:
            加载的模型实例，失败返回None
        """
        if not self._init_distributed():
            return None
        
        # 获取配置并动态设置window_size
        cfg = WAN_CONFIGS[self.task]
        
        # 根据size动态计算最优window_size
        optimal_window = get_optimal_window_size(size)
        cfg.window_size = optimal_window
        
        logger.info(f"Loading model with config: task={self.task}, size={size}, window_size={cfg.window_size}")
        
        try:
            # 创建模型实例(在主进程rank 0上)
            if self.rank == 0:
                shard_fn = None
                
                # 如果使用FSDP，需要定义shard函数
                if t5_fsdp or dit_fsdp:
                    from wan.distributed.fsdp import shard_model
                    device_id = int(os.getenv("LOCAL_RANK", 0))
                    shard_fn = partial(shard_model, device_id=device_id)
                
                self.model = WanI2V_any(
                    config=cfg,
                    checkpoint_dir=self.checkpoint_dir,
                    device_id=0,  # 主进程使用GPU 0
                    rank=0,
                    t5_fsdp=t5_fsdp,
                    dit_fsdp=dit_fsdp,
                    use_usp=False,
                    t5_cpu=t5_cpu,
                    init_on_cpu=True,
                )
                
                logger.info("Model loaded successfully on rank 0")
                
                # 同步所有进程
                if dist.is_initialized():
                    dist.barrier()
                    
            else:
                # 其他进程等待主进程加载完成
                if dist.is_initialized():
                    dist.barrier()
                    
            self.is_initialized = True
            return self.model
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def cleanup(self):
        """清理分布式环境"""
        if dist.is_initialized():
            dist.destroy_process_group()
            logger.info("Distributed process group destroyed")
        
        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    def __del__(self):
        """析构函数，确保清理"""
        self.cleanup()


class SingleModelLoader:
    """单GPU模型加载器 - 用于非分布式场景"""

    def __init__(
        self,
        checkpoint_dir: str = "Wan2.1-I2V-14B-480P",
        task: str = "i2v-14B",
    ):
        """初始化单GPU模型加载器

        Args:
            checkpoint_dir: 模型检查点目录
            task: 任务类型
        """
        self.checkpoint_dir = checkpoint_dir
        self.task = task

    def load_model(
        self,
        size: str = "720x1280",
    ) -> Optional[WanI2V_any]:
        """加载模型(单GPU)

        Args:
            size: 视频分辨率

        Returns:
            加载的模型实例，失败返回None
        """

        # 获取配置并动态设置window_size
        cfg = WAN_CONFIGS[self.task]

        # 根据size动态计算最优window_size
        optimal_window = get_optimal_window_size(size)
        cfg.window_size = optimal_window

        logger.info(f"Loading model with config: task={self.task}, size={size}, window_size={cfg.window_size}")

        try:
            from wan.distributed.fsdp import shard_model
            from functools import partial

            shard_fn = partial(shard_model, device_id=0)

            self.model = WanI2V_any(
                config=cfg,
                checkpoint_dir=self.checkpoint_dir,
                device_id=0,
                rank=0,
                t5_fsdp=False,  # 单GPU不使用FSDP
                dit_fsdp=False,
                use_usp=False,
                t5_cpu=False,
                init_on_cpu=True,
            )

            logger.info("Model loaded successfully (single GPU)")
            return self.model

        except Exception as e:
            logger.error(f"Failed to load model (single GPU): {e}")
            import traceback
            traceback.print_exc()
            return None


def create_model_loader(
    checkpoint_dir: str,
    task: str = "i2v-14B",
    distributed: bool = True,
    world_size: int = 8,
) -> Any:
    """创建模型加载器工厂函数
    
    Args:
       checkpoint_dir: 模型检查点目录 
       task: 任务类型  
       distributed: 是否使用分布式(8卡)
       world_size: GPU数量 
       
    Returns:
       模型加载器实例 
    """
    if distributed and world_size > 1:
         return DistributedModelLoader(
             checkpoint_dir=checkpoint_dir, 
             task=task, 
             world_size=world_size 
         )
    else:
         return SingleModelLoader(
             checkpoint_dir=checkpoint_dir, 
             task=task 
         )
