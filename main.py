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

# 导入配置系统
from config import get_config, diagnose_config

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

    logger.info("端到端音频处理系统 (知识库集成版) - 统一配置系统")
    logger.info("流程: MP3音频 → WAV转换 → 说话人分离 → 切分子音频 → ASR语音识别 → [异步]Gleaning多轮清洗 → 问答对抽取 → 高质量知识库")
    logger.info("=" * 100)

    # 显示配置信息（调试模式）
    if get_config('system.logging.level') == 'DEBUG':
        logger.info("配置系统诊断信息:")
        diagnose_config()

    # 创建音频处理器（使用配置系统的默认值）
    processor = AudioProcessor()

    # 记录实际使用的配置
    logger.info(f"配置信息:")
    logger.info(f"  - 异步LLM处理: {processor.enable_async_llm}")
    logger.info(f"  - 最大并发LLM任务: {processor.max_concurrent_llm}")
    logger.info(f"  - 知识库集成: {processor.enable_knowledge_base}")
    logger.info(f"  - 自动清理: {processor.enable_auto_cleanup}")
    logger.info(f"  - Gleaning多轮清洗: {processor.enable_gleaning} (最大轮数: {processor.max_gleaning_rounds})")
    logger.info(f"  - 设备: {get_config('system.device.cuda_device')}")
    logger.info(f"  - 环境: {get_config('_environment', '未知')}")
    logger.info("=" * 100)

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

        # 显示知识库统计信息
        if processor.enable_knowledge_base and processor.knowledge_processor:
            logger.info("\n" + "=" * 60)
            logger.info("📊 知识库统计信息")
            logger.info("=" * 60)

            try:
                kb_status = processor.knowledge_processor.get_knowledge_base_status()
                kb_stats = kb_status.get('knowledge_base', {})
                processing_stats = kb_status.get('processing', {})

                logger.info(f"总问答对数量: {kb_stats.get('total_qa_pairs', 0)}")
                logger.info(f"活跃缓冲区: {kb_stats.get('current_active_buffer', 'unknown')} (大小: {kb_stats.get('active_buffer_size', 0)})")
                logger.info(f"非活跃缓冲区大小: {kb_stats.get('inactive_buffer_size', 0)}")
                logger.info(f"处理文件总数: {processing_stats.get('total_files_processed', 0)}")
                logger.info(f"抽取成功: {processing_stats.get('qa_extraction_success', 0)}, 失败: {processing_stats.get('qa_extraction_failed', 0)}")
                logger.info(f"总抽取问答对: {processing_stats.get('total_qa_pairs_extracted', 0)}")

                # 压缩统计
                compaction_stats = kb_status.get('compaction', {})
                if compaction_stats:
                    logger.info(f"压缩次数: {compaction_stats.get('total_compactions', 0)}")
                    compression_ratio = compaction_stats.get('compression_ratio', 0)
                    logger.info(f"最近压缩比例: {compression_ratio:.2%}")

                try:
                    from config import get_config
                    knowledge_base_file = get_config('system.paths.knowledge_base_file', 'data/output/knowledgeDatabase.md')
                except Exception:
                    knowledge_base_file = "data/output/knowledgeDatabase.md"
                logger.info(f"知识库文件: {knowledge_base_file}")
                logger.info("=" * 60)

            except Exception as e:
                logger.error(f"获取知识库统计失败: {str(e)}")

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