"""
知识库集成模块
将问答对抽取集成到现有音频处理流程中
"""

import os
from typing import Dict, Any, Optional
from datetime import datetime
import threading

from src.utils.logger import get_logger
from src.core.knowledge_base import get_knowledge_base, ProcessingStatus
from src.core.qa_extractor import QAExtractor
from src.core.qa_compactor import get_qa_compactor, CompactionScheduler
from src.core.system_monitor import get_system_monitor, cleanup_system_monitor
from src.utils.concurrency import get_concurrency_monitor, thread_safe_operation


class KnowledgeProcessor:
    """知识处理器 - 管理问答对抽取和知识库集成"""

    def __init__(self, enable_auto_qa_extraction: bool = True, enable_auto_compaction: bool = True, enable_system_monitoring: bool = True):
        """
        初始化知识处理器

        Args:
            enable_auto_qa_extraction: 是否启用自动问答对抽取
            enable_auto_compaction: 是否启用自动压缩
            enable_system_monitoring: 是否启用系统监控
        """
        self.logger = get_logger(__name__)
        self.enable_auto_qa_extraction = enable_auto_qa_extraction
        self.enable_auto_compaction = enable_auto_compaction
        self.enable_system_monitoring = enable_system_monitoring

        # 核心组件
        self.knowledge_base = get_knowledge_base()
        self.qa_extractor = None  # 延迟初始化
        self.qa_compactor = None  # 延迟初始化
        self.compaction_scheduler = None  # 延迟初始化
        self.system_monitor = None  # 延迟初始化

        # 并发监控
        self.concurrency_monitor = get_concurrency_monitor()

        # 处理统计
        self.processing_stats = {
            'total_files_processed': 0,
            'qa_extraction_success': 0,
            'qa_extraction_failed': 0,
            'total_qa_pairs_extracted': 0,
            'last_processing_time': None
        }

        # 初始化系统监控
        if self.enable_system_monitoring:
            self.system_monitor = get_system_monitor()
            self.system_monitor.start_monitoring()
            self.logger.info("系统监控已启动")

        # 检查是否需要立即初始化压缩器
        if self.enable_auto_compaction:
            self._check_and_initialize_compactor()

        self.logger.info(f"知识处理器初始化完成 - QA抽取: {enable_auto_qa_extraction}, 自动压缩: {enable_auto_compaction}, 系统监控: {enable_system_monitoring}")

    def _initialize_qa_extractor(self):
        """延迟初始化问答对抽取器"""
        if self.qa_extractor is None:
            try:
                self.qa_extractor = QAExtractor()
                self.logger.info("问答对抽取器初始化成功")
            except Exception as e:
                self.logger.warning(f"问答对抽取器初始化失败: {str(e)}")
                self.qa_extractor = False  # 标记为失败

    def _check_and_initialize_compactor(self):
        """检查知识库大小并在需要时初始化压缩器"""
        if not self.enable_auto_compaction:
            return

        try:
            kb_stats = self.knowledge_base.get_statistics()
            total_qa_pairs = kb_stats.get('total_qa_pairs', 0)

            # 如果问答对数量超过阈值且压缩器未初始化，则立即初始化
            if total_qa_pairs >= 50 and self.qa_compactor is None:
                self.logger.info(f"知识库问答对数量({total_qa_pairs})已超过阈值，启动压缩器")
                self._initialize_qa_compactor()
            elif total_qa_pairs < 50:
                self.logger.debug(f"知识库问答对数量({total_qa_pairs})未达到压缩阈值(50)")

        except Exception as e:
            self.logger.warning(f"检查压缩器初始化条件失败: {str(e)}")

    def _initialize_qa_compactor(self):
        """延迟初始化问答对压缩器"""
        if self.qa_compactor is None:
            try:
                self.qa_compactor = get_qa_compactor()
                self.logger.info("问答对压缩器初始化成功")

                # 启动压缩调度器
                if self.enable_auto_compaction:
                    self.compaction_scheduler = CompactionScheduler(self.qa_compactor, interval_minutes=1)
                    self.compaction_scheduler.start_scheduler()
                    self.logger.info("自动压缩调度器启动成功")

            except Exception as e:
                self.logger.warning(f"问答对压缩器初始化失败: {str(e)}")
                self.qa_compactor = False  # 标记为失败

    @thread_safe_operation("process_cleaned_file")
    def process_cleaned_file(self, file_path: str, force_extraction: bool = False) -> Dict[str, Any]:
        """
        处理已清洗的文件，执行问答对抽取

        Args:
            file_path: 已清洗的文件路径
            force_extraction: 是否强制重新抽取

        Returns:
            Dict[str, Any]: 处理结果
        """
        if not self.enable_auto_qa_extraction:
            return {
                "success": False,
                "message": "自动问答对抽取已禁用"
            }

        try:
            # 检查文件是否存在
            if not os.path.exists(file_path):
                return {
                    "success": False,
                    "error": f"文件不存在: {file_path}"
                }

            # 检查文件状态
            file_status = self.knowledge_base.get_file_status(file_path)
            if not force_extraction and file_status and file_status.status in [
                ProcessingStatus.QA_EXTRACTED, ProcessingStatus.COMPACTED
            ]:
                self.logger.info(f"文件已处理，跳过: {file_path} (状态: {file_status.status.value})")
                return {
                    "success": True,
                    "message": "文件已处理，跳过",
                    "file_path": file_path,
                    "status": file_status.status.value
                }

            # 更新文件状态为清洗完成
            self.knowledge_base.update_file_status(
                file_path,
                ProcessingStatus.CLEAN_FINISHED,
                {"clean_finished_time": datetime.now().isoformat()}
            )

            # 初始化问答对抽取器
            self._initialize_qa_extractor()
            if not self.qa_extractor or self.qa_extractor is False:
                return {
                    "success": False,
                    "error": "问答对抽取器不可用",
                    "file_path": file_path
                }

            # 执行问答对抽取
            self.logger.info(f"开始抽取文件: {os.path.basename(file_path)}")
            extraction_result = self.qa_extractor.extract_and_save_qa_pairs(file_path)

            # 更新统计信息
            self.processing_stats['total_files_processed'] += 1
            self.processing_stats['last_processing_time'] = datetime.now().isoformat()

            if extraction_result["success"]:
                qa_count = extraction_result.get("qa_count", 0)
                self.processing_stats['qa_extraction_success'] += 1
                self.processing_stats['total_qa_pairs_extracted'] += qa_count

                self.logger.info(f"✅ 文件抽取成功: {os.path.basename(file_path)}, 抽取 {qa_count} 个问答对")

                # 检查是否需要初始化压缩器
                if self.enable_auto_compaction and self.qa_compactor is None:
                    self._check_and_initialize_compactor()

                return {
                    "success": True,
                    "file_path": file_path,
                    "qa_count": qa_count,
                    "summary": extraction_result.get("summary", {}),
                    "token_usage": extraction_result.get("token_usage", {})
                }
            else:
                self.processing_stats['qa_extraction_failed'] += 1
                error_msg = extraction_result.get("error", "未知错误")

                self.logger.error(f"❌ 文件处理失败: {os.path.basename(file_path)}, 错误: {error_msg}")

                return {
                    "success": False,
                    "error": error_msg,
                    "file_path": file_path
                }

        except Exception as e:
            self.processing_stats['qa_extraction_failed'] += 1
            self.logger.error(f"处理文件异常: {file_path}, 错误: {str(e)}")

            return {
                "success": False,
                "error": str(e),
                "file_path": file_path
            }

    @thread_safe_operation("batch_process_cleaned_files")
    def batch_process_cleaned_files(self) -> Dict[str, Any]:
        """
        批量处理所有已清洗的文件

        Returns:
            Dict[str, Any]: 批量处理结果
        """
        if not self.enable_auto_qa_extraction:
            return {
                "success": False,
                "message": "自动问答对抽取已禁用"
            }

        try:
            # 获取所有清洗完成但未抽取问答对的文件
            clean_finished_files = self.knowledge_base.get_clean_finished_files()

            if not clean_finished_files:
                self.logger.info("没有找到需要处理的文件")
                return {
                    "success": True,
                    "message": "没有需要处理的文件",
                    "processed_files": 0
                }

            self.logger.info(f"开始批量处理 {len(clean_finished_files)} 个文件")

            results = []
            success_count = 0
            error_count = 0

            for file_path in clean_finished_files:
                result = self.process_cleaned_file(file_path)
                results.append(result)

                if result["success"]:
                    success_count += 1
                else:
                    error_count += 1

            self.logger.info(f"批量处理完成: 成功 {success_count}, 失败 {error_count}")

            return {
                "success": True,
                "total_files": len(clean_finished_files),
                "success_count": success_count,
                "error_count": error_count,
                "results": results,
                "processing_stats": self.processing_stats.copy()
            }

        except Exception as e:
            self.logger.error(f"批量处理异常: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    @thread_safe_operation("trigger_compaction")
    def trigger_compaction(self, similarity_threshold: float = 0.75) -> Dict[str, Any]:
        """
        手动触发压缩操作

        Args:
            similarity_threshold: 相似度阈值

        Returns:
            Dict[str, Any]: 压缩结果
        """
        try:
            # 初始化压缩器
            self._initialize_qa_compactor()
            if not self.qa_compactor or self.qa_compactor is False:
                return {
                    "success": False,
                    "error": "问答对压缩器不可用"
                }

            # 获取当前的问答对数据
            all_qa_pairs = self.knowledge_base.get_all_qa_pairs()

            if len(all_qa_pairs) < 10:  # 至少需要10个问答对才有压缩意义
                return {
                    "success": False,
                    "message": f"问答对数量不足，无需压缩 (当前: {len(all_qa_pairs)})"
                }

            self.logger.info(f"开始压缩 {len(all_qa_pairs)} 个问答对...")

            # 创建快照
            snapshot = self.knowledge_base.create_snapshot()
            if not snapshot:
                return {
                    "success": False,
                    "error": "创建快照失败"
                }

            # 执行压缩（默认使用LLM智能检验）
            compaction_result = self.qa_compactor.compact_qa_pairs(
                snapshot.data, similarity_threshold, use_llm_similarity=True
            )

            if compaction_result["success"]:
                # 切换缓冲区并同步尾部数据
                compacted_qa_pairs = compaction_result["compacted_qa_pairs"]
                switch_success = self.knowledge_base.switch_buffers_with_tail_sync(compacted_qa_pairs)

                if switch_success:
                    self.logger.info(f"✅ 压缩完成: {compaction_result['compression_ratio']:.2%} 压缩率")

                    return {
                        "success": True,
                        "original_count": compaction_result["original_count"],
                        "final_count": compaction_result["final_count"],
                        "compression_ratio": compaction_result["compression_ratio"],
                        "processing_time": compaction_result["processing_time"]
                    }
                else:
                    return {
                        "success": False,
                        "error": "缓冲区切换失败"
                    }
            else:
                return {
                    "success": False,
                    "error": compaction_result.get("error", "压缩失败")
                }

        except Exception as e:
            self.logger.error(f"压缩操作异常: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    def get_knowledge_base_status(self) -> Dict[str, Any]:
        """
        获取知识库状态

        Returns:
            Dict[str, Any]: 知识库状态信息
        """
        try:
            # 知识库统计
            kb_stats = self.knowledge_base.get_statistics()

            # 处理统计
            processing_stats = self.processing_stats.copy()

            # 并发统计
            concurrency_stats = self.concurrency_monitor.get_statistics()

            # 压缩统计
            compaction_stats = {}
            scheduler_stats = {}
            if self.qa_compactor and self.qa_compactor is not False:
                compaction_stats = self.qa_compactor.get_compaction_statistics()
            if self.compaction_scheduler:
                scheduler_stats = self.compaction_scheduler.get_scheduler_statistics()

            # 系统监控统计
            system_monitor_stats = {}
            if self.system_monitor:
                system_monitor_stats = self.system_monitor.get_system_status()

            return {
                "knowledge_base": kb_stats,
                "processing": processing_stats,
                "concurrency": concurrency_stats,
                "compaction": compaction_stats,
                "scheduler": scheduler_stats,
                "system_monitor": system_monitor_stats,
                "auto_qa_extraction": self.enable_auto_qa_extraction,
                "auto_compaction": self.enable_auto_compaction,
                "system_monitoring": self.enable_system_monitoring,
                "components_status": {
                    "qa_extractor": self.qa_extractor is not None and self.qa_extractor is not False,
                    "qa_compactor": self.qa_compactor is not None and self.qa_compactor is not False,
                    "compaction_scheduler": self.compaction_scheduler is not None and self.compaction_scheduler.is_running,
                    "system_monitor": self.system_monitor is not None and self.system_monitor.is_monitoring
                }
            }

        except Exception as e:
            self.logger.error(f"获取知识库状态失败: {str(e)}")
            return {
                "error": str(e),
                "components_status": {
                    "qa_extractor": False,
                    "qa_compactor": False,
                    "compaction_scheduler": False
                }
            }

    def shutdown(self):
        """关闭知识处理器，清理资源"""
        try:
            # 停止压缩调度器
            if self.compaction_scheduler:
                self.compaction_scheduler.stop_scheduler()

            # 停止系统监控
            if self.system_monitor:
                self.system_monitor.stop_monitoring()

            # 清理知识库
            self.knowledge_base.cleanup()

            self.logger.info("知识处理器已关闭")

        except Exception as e:
            self.logger.error(f"知识处理器关闭失败: {str(e)}")


# 全局知识处理器实例
_knowledge_processor: Optional[KnowledgeProcessor] = None
_processor_lock = threading.Lock()


def get_knowledge_processor(enable_auto_qa_extraction: bool = True, enable_auto_compaction: bool = True, enable_system_monitoring: bool = True) -> KnowledgeProcessor:
    """
    获取全局知识处理器实例

    Args:
        enable_auto_qa_extraction: 是否启用自动问答对抽取
        enable_auto_compaction: 是否启用自动压缩
        enable_system_monitoring: 是否启用系统监控

    Returns:
        KnowledgeProcessor: 知识处理器实例
    """
    global _knowledge_processor

    if _knowledge_processor is None:
        with _processor_lock:
            if _knowledge_processor is None:
                _knowledge_processor = KnowledgeProcessor(
                    enable_auto_qa_extraction=enable_auto_qa_extraction,
                    enable_auto_compaction=enable_auto_compaction,
                    enable_system_monitoring=enable_system_monitoring
                )

    return _knowledge_processor


def cleanup_knowledge_processor():
    """清理全局知识处理器实例"""
    global _knowledge_processor

    if _knowledge_processor is not None:
        _knowledge_processor.shutdown()
        _knowledge_processor = None

    # 清理全局监控器
    cleanup_system_monitor()