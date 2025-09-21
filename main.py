#!/usr/bin/env python3
"""
端到端音频处理系统
根目录主入口文件 - 重构后的项目结构
"""

import sys
import os
import warnings

# 将项目根目录添加到Python路径中
sys.path.insert(0, os.path.dirname(__file__))

# 过滤烦人的警告信息
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
warnings.filterwarnings("ignore", category=UserWarning, module="speechbrain")
warnings.filterwarnings("ignore", message=".*torchaudio._backend.*deprecated.*")
warnings.filterwarnings("ignore", message=".*TensorFloat-32.*")
warnings.filterwarnings("ignore", message=".*std().*degrees of freedom.*")

# 设置环境变量减少一些库的输出
os.environ['PYTHONWARNINGS'] = 'ignore'

from src.utils.processor import AudioProcessor
from src.utils.logger import get_logger
import signal


def signal_handler(sig, frame):
    """处理中断信号"""
    _ = sig, frame  # 避免未使用警告
    logger = get_logger(__name__)
    logger.info('\n收到中断信号，正在优雅关闭处理器...')
    # 清理逻辑在main函数的finally块中执行
    sys.exit(0)


def main():
    """主程序入口"""
    logger = get_logger(__name__)

    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logger.info("端到端音频处理系统 (异步优化版)")
    logger.info("流程: MP3音频 → WAV转换 → 说话人分离 → 切分子音频 → ASR语音识别 → [异步]Gleaning多轮清洗 → 高质量知识库语料")
    logger.info("=" * 100)

    # 创建音频处理器（启用异步LLM）
    processor = AudioProcessor(
        enable_async_llm=True,      # 启用异步LLM处理
        max_concurrent_llm=4        # 最大并发LLM任务数
    )

    try:
        # 执行批量处理（包含MP3转WAV预处理）
        result = processor.process_batch()

        logger.info(f"\n主流程处理完成！成功: {result['success']}个, 失败: {result['error']}个, 跳过: {result['skipped']}个")

        # 如果有异步LLM任务，等待完成
        if result.get('async_llm_tasks', 0) > 0:
            logger.info(f"\n检测到 {result['async_llm_tasks']} 个异步LLM任务正在后台处理...")
            logger.info("等待所有异步LLM任务完成...")
            wait_result = processor.wait_for_async_llm_tasks()  # 无超时限制，等待直到完成
            if wait_result['status'] == 'completed':
                logger.info("所有异步LLM任务已完成！")
            else:
                logger.warning(f"等待结束，状态: {wait_result['status']}")

    except KeyboardInterrupt:
        logger.info("\n收到中断信号，正在清理...")
    except Exception as e:
        logger.error(f"\n程序执行出错: {str(e)}")
    finally:
        # 清理资源
        logger.info("正在关闭处理器...")
        processor.shutdown()
        logger.info("处理器已关闭，程序结束")


if __name__ == "__main__":
    main()