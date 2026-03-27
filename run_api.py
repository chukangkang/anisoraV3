# -*- coding: utf-8 -*-
"""
AnisoraV3 API服务入口 - 启动FastAPI服务器
"""
import os
import sys
import logging

# 设置环境变量
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """主入口函数"""
    import uvicorn
    from api.config import API_CONFIG
    
    logger.info("=" * 50)
    logger.info("Starting AnisoraV3 Video Generation API Server")
    logger.info("=" * 50)
    logger.info(f"Host: {API_CONFIG.host}")
    logger.info(f"Port: {API_CONFIG.port}")
    logger.info(f"Workers: {API_CONFIG.workers}")
    logger.info(f"Model: Wan2.1-I2V-14B")
    logger.info(f"Distributed: 8-GPU")
    logger.info("=" * 50)
    
    # 启动服务器
    uvicorn.run(
        "api.routes:app",
        host=API_CONFIG.host,
        port=API_CONFIG.port,
        workers=API_CONFIG.workers,
        log_level="info",
        reload=False,  # 生产环境不使用reload
    )


if __name__ == "__main__":
    main()
