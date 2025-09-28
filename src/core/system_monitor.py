"""
系统监控模块
监控知识库系统的运行状态、性能指标和资源使用情况
"""

import os
import threading
import time
import psutil
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import json

from src.utils.logger import get_logger
from src.utils.concurrency import get_concurrency_monitor
from src.core.knowledge_base import get_knowledge_base
from src.core.qa_compactor import get_qa_compactor


@dataclass
class SystemMetrics:
    """系统指标"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_usage_percent: float
    disk_free_gb: float
    process_count: int
    thread_count: int
    file_descriptors: int


@dataclass
class KnowledgeBaseMetrics:
    """知识库指标"""
    timestamp: datetime
    total_qa_pairs: int
    active_buffer_size: int
    inactive_buffer_size: int
    clean_finished_files: int
    qa_extracted_files: int
    processing_success_rate: float
    avg_qa_pairs_per_file: float
    last_compaction_time: Optional[str]
    compression_ratio: float


@dataclass
class PerformanceMetrics:
    """性能指标"""
    timestamp: datetime
    avg_processing_time_per_file: float
    avg_qa_extraction_time: float
    avg_compaction_time: float
    total_files_processed: int
    total_tokens_consumed: int
    processing_throughput: float  # files per hour


class SystemMonitor:
    """系统监控器"""

    def __init__(self, monitoring_interval: int = 60, retention_hours: int = 24):
        """
        初始化系统监控器

        Args:
            monitoring_interval: 监控间隔（秒）
            retention_hours: 数据保留时间（小时）
        """
        self.logger = get_logger(__name__)
        self.monitoring_interval = monitoring_interval
        self.retention_hours = retention_hours

        # 监控状态
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()

        # 数据存储
        self.system_metrics: List[SystemMetrics] = []
        self.kb_metrics: List[KnowledgeBaseMetrics] = []
        self.performance_metrics: List[PerformanceMetrics] = []
        self.metrics_lock = threading.RLock()

        # 警报配置
        self.alert_thresholds = {
            'cpu_percent': 80.0,
            'memory_percent': 85.0,
            'disk_usage_percent': 90.0,
            'processing_error_rate': 0.2,  # 20%
            'qa_extraction_error_rate': 0.15,  # 15%
        }

        # 警报状态
        self.active_alerts: Dict[str, Dict[str, Any]] = {}
        self.alert_history: List[Dict[str, Any]] = []

        # 组件引用
        self.knowledge_base = get_knowledge_base()
        self.concurrency_monitor = get_concurrency_monitor()
        self.qa_compactor = None  # 延迟初始化

        self.logger.info(f"系统监控器初始化完成 - 监控间隔: {monitoring_interval}s, 数据保留: {retention_hours}h")

    def start_monitoring(self):
        """启动系统监控"""
        if self.is_monitoring:
            self.logger.warning("系统监控已在运行")
            return

        self.is_monitoring = True
        self.stop_event.clear()
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()

        self.logger.info("系统监控启动成功")

    def stop_monitoring(self):
        """停止系统监控"""
        if not self.is_monitoring:
            return

        self.is_monitoring = False
        self.stop_event.set()

        if self.monitor_thread:
            try:
                from config import get_config
                timeout = get_config('system.endpoints.network.monitor_thread_timeout', 30)
            except Exception:
                timeout = 30
            self.monitor_thread.join(timeout=timeout)

        self.logger.info("系统监控已停止")

    def _monitoring_loop(self):
        """监控循环"""
        while self.is_monitoring and not self.stop_event.is_set():
            try:
                # 收集系统指标
                self._collect_system_metrics()

                # 收集知识库指标
                self._collect_knowledge_base_metrics()

                # 收集性能指标
                self._collect_performance_metrics()

                # 检查警报
                self._check_alerts()

                # 清理过期数据
                self._cleanup_old_metrics()

                # 等待下个监控周期
                if self.stop_event.wait(timeout=self.monitoring_interval):
                    break

            except Exception as e:
                self.logger.error(f"监控循环异常: {str(e)}")

    def _collect_system_metrics(self):
        """收集系统指标"""
        try:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=1)

            # 内存使用情况
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_mb = memory.used / (1024 * 1024)
            memory_available_mb = memory.available / (1024 * 1024)

            # 磁盘使用情况
            disk = psutil.disk_usage('/')
            disk_usage_percent = (disk.used / disk.total) * 100
            disk_free_gb = disk.free / (1024 * 1024 * 1024)

            # 进程信息
            current_process = psutil.Process()
            process_count = len(psutil.pids())
            thread_count = current_process.num_threads()

            # 文件描述符
            try:
                file_descriptors = current_process.num_fds()
            except (AttributeError, psutil.AccessDenied):
                file_descriptors = 0

            # 创建指标
            metrics = SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_used_mb=memory_used_mb,
                memory_available_mb=memory_available_mb,
                disk_usage_percent=disk_usage_percent,
                disk_free_gb=disk_free_gb,
                process_count=process_count,
                thread_count=thread_count,
                file_descriptors=file_descriptors
            )

            # 存储指标
            with self.metrics_lock:
                self.system_metrics.append(metrics)

        except Exception as e:
            self.logger.error(f"收集系统指标失败: {str(e)}")

    def _collect_knowledge_base_metrics(self):
        """收集知识库指标"""
        try:
            # 获取知识库统计
            kb_stats = self.knowledge_base.get_statistics()

            # 获取文件状态统计
            clean_finished_count = len(self.knowledge_base.get_clean_finished_files())
            total_files = len(self.knowledge_base.file_status_map)

            # 计算成功率
            processing_success_rate = 0.0
            avg_qa_pairs_per_file = 0.0

            if total_files > 0:
                success_files = 0
                total_qa_pairs = 0

                for file_status in self.knowledge_base.file_status_map.values():
                    if hasattr(file_status, 'metadata') and file_status.metadata:
                        qa_count = file_status.metadata.get('qa_count', 0)
                        if qa_count > 0:
                            success_files += 1
                            total_qa_pairs += qa_count

                processing_success_rate = success_files / total_files if total_files > 0 else 0.0
                avg_qa_pairs_per_file = total_qa_pairs / success_files if success_files > 0 else 0.0

            # 获取压缩信息
            compression_ratio = 0.0
            last_compaction_time = None

            if not self.qa_compactor:
                try:
                    self.qa_compactor = get_qa_compactor()
                except:
                    pass

            if self.qa_compactor:
                compaction_stats = self.qa_compactor.get_compaction_statistics()
                compression_ratio = compaction_stats.get('compression_ratio', 0.0)
                last_compaction_time = compaction_stats.get('last_compaction_time')

            # 创建指标
            metrics = KnowledgeBaseMetrics(
                timestamp=datetime.now(),
                total_qa_pairs=kb_stats.get('total_qa_pairs', 0),
                active_buffer_size=kb_stats.get('active_buffer_size', 0),
                inactive_buffer_size=kb_stats.get('inactive_buffer_size', 0),
                clean_finished_files=clean_finished_count,
                qa_extracted_files=kb_stats.get('clean_finished_files', 0),
                processing_success_rate=processing_success_rate,
                avg_qa_pairs_per_file=avg_qa_pairs_per_file,
                last_compaction_time=last_compaction_time,
                compression_ratio=compression_ratio
            )

            # 存储指标
            with self.metrics_lock:
                self.kb_metrics.append(metrics)

        except Exception as e:
            self.logger.error(f"收集知识库指标失败: {str(e)}")

    def _collect_performance_metrics(self):
        """收集性能指标"""
        try:
            # 获取并发统计
            concurrency_stats = self.concurrency_monitor.get_statistics()

            # 计算平均处理时间（简化实现）
            avg_processing_time_per_file = 0.0
            avg_qa_extraction_time = 0.0
            avg_compaction_time = 0.0
            processing_throughput = 0.0

            # 这里可以基于历史数据计算更准确的性能指标
            # 简化实现，使用固定值
            total_files_processed = concurrency_stats.get('total_operations', 0)
            total_tokens_consumed = 0  # 需要从实际统计中获取

            # 计算吞吐量（文件/小时）
            if len(self.performance_metrics) > 0:
                time_diff = (datetime.now() - self.performance_metrics[0].timestamp).total_seconds() / 3600
                if time_diff > 0:
                    processing_throughput = total_files_processed / time_diff

            # 创建指标
            metrics = PerformanceMetrics(
                timestamp=datetime.now(),
                avg_processing_time_per_file=avg_processing_time_per_file,
                avg_qa_extraction_time=avg_qa_extraction_time,
                avg_compaction_time=avg_compaction_time,
                total_files_processed=total_files_processed,
                total_tokens_consumed=total_tokens_consumed,
                processing_throughput=processing_throughput
            )

            # 存储指标
            with self.metrics_lock:
                self.performance_metrics.append(metrics)

        except Exception as e:
            self.logger.error(f"收集性能指标失败: {str(e)}")

    def _check_alerts(self):
        """检查警报条件"""
        try:
            current_time = datetime.now()

            # 获取最新的系统指标
            if self.system_metrics:
                latest_system = self.system_metrics[-1]

                # CPU使用率警报
                self._check_threshold_alert(
                    'cpu_high',
                    latest_system.cpu_percent,
                    self.alert_thresholds['cpu_percent'],
                    f"CPU使用率过高: {latest_system.cpu_percent:.1f}%"
                )

                # 内存使用率警报
                self._check_threshold_alert(
                    'memory_high',
                    latest_system.memory_percent,
                    self.alert_thresholds['memory_percent'],
                    f"内存使用率过高: {latest_system.memory_percent:.1f}%"
                )

                # 磁盘使用率警报
                self._check_threshold_alert(
                    'disk_high',
                    latest_system.disk_usage_percent,
                    self.alert_thresholds['disk_usage_percent'],
                    f"磁盘使用率过高: {latest_system.disk_usage_percent:.1f}%"
                )

            # 获取最新的知识库指标
            if self.kb_metrics:
                latest_kb = self.kb_metrics[-1]

                # 处理错误率警报
                error_rate = 1.0 - latest_kb.processing_success_rate
                self._check_threshold_alert(
                    'processing_error_high',
                    error_rate,
                    self.alert_thresholds['processing_error_rate'],
                    f"处理错误率过高: {error_rate:.1%}"
                )

        except Exception as e:
            self.logger.error(f"检查警报失败: {str(e)}")

    def _check_threshold_alert(self, alert_id: str, current_value: float, threshold: float, message: str):
        """检查阈值警报"""
        if current_value >= threshold:
            if alert_id not in self.active_alerts:
                # 新警报
                alert = {
                    'id': alert_id,
                    'message': message,
                    'start_time': datetime.now(),
                    'current_value': current_value,
                    'threshold': threshold,
                    'severity': 'high' if current_value >= threshold * 1.1 else 'medium'
                }
                self.active_alerts[alert_id] = alert
                self.alert_history.append(alert.copy())
                self.logger.warning(f"🚨 警报触发: {message}")
            else:
                # 更新现有警报
                self.active_alerts[alert_id]['current_value'] = current_value
        else:
            if alert_id in self.active_alerts:
                # 警报解除
                alert = self.active_alerts.pop(alert_id)
                duration = (datetime.now() - alert['start_time']).total_seconds()
                self.logger.info(f"✅ 警报解除: {alert['message']} (持续: {duration:.0f}秒)")

    def _cleanup_old_metrics(self):
        """清理过期的指标数据"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=self.retention_hours)

            with self.metrics_lock:
                # 清理系统指标
                self.system_metrics = [m for m in self.system_metrics if m.timestamp > cutoff_time]

                # 清理知识库指标
                self.kb_metrics = [m for m in self.kb_metrics if m.timestamp > cutoff_time]

                # 清理性能指标
                self.performance_metrics = [m for m in self.performance_metrics if m.timestamp > cutoff_time]

            # 清理警报历史
            self.alert_history = [a for a in self.alert_history if a['start_time'] > cutoff_time]

        except Exception as e:
            self.logger.error(f"清理过期指标失败: {str(e)}")

    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态摘要"""
        try:
            with self.metrics_lock:
                latest_system = self.system_metrics[-1] if self.system_metrics else None
                latest_kb = self.kb_metrics[-1] if self.kb_metrics else None
                latest_perf = self.performance_metrics[-1] if self.performance_metrics else None

            status = {
                'monitoring': {
                    'is_active': self.is_monitoring,
                    'interval_seconds': self.monitoring_interval,
                    'data_retention_hours': self.retention_hours,
                    'last_update': datetime.now().isoformat()
                },
                'system': asdict(latest_system) if latest_system else None,
                'knowledge_base': asdict(latest_kb) if latest_kb else None,
                'performance': asdict(latest_perf) if latest_perf else None,
                'alerts': {
                    'active_count': len(self.active_alerts),
                    'active_alerts': list(self.active_alerts.values()),
                    'recent_alerts': self.alert_history[-10:]  # 最近10个警报
                },
                'metrics_count': {
                    'system': len(self.system_metrics),
                    'knowledge_base': len(self.kb_metrics),
                    'performance': len(self.performance_metrics)
                }
            }

            return status

        except Exception as e:
            self.logger.error(f"获取系统状态失败: {str(e)}")
            return {
                'error': str(e),
                'monitoring': {'is_active': False}
            }

    def get_historical_metrics(self, metric_type: str, hours: int = 1) -> List[Dict[str, Any]]:
        """
        获取历史指标数据

        Args:
            metric_type: 指标类型 ('system', 'knowledge_base', 'performance')
            hours: 时间范围（小时）

        Returns:
            List[Dict[str, Any]]: 历史指标数据
        """
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)

            with self.metrics_lock:
                if metric_type == 'system':
                    metrics = [asdict(m) for m in self.system_metrics if m.timestamp > cutoff_time]
                elif metric_type == 'knowledge_base':
                    metrics = [asdict(m) for m in self.kb_metrics if m.timestamp > cutoff_time]
                elif metric_type == 'performance':
                    metrics = [asdict(m) for m in self.performance_metrics if m.timestamp > cutoff_time]
                else:
                    return []

            # 转换datetime为字符串
            for metric in metrics:
                if 'timestamp' in metric and isinstance(metric['timestamp'], datetime):
                    metric['timestamp'] = metric['timestamp'].isoformat()

            return metrics

        except Exception as e:
            self.logger.error(f"获取历史指标失败: {str(e)}")
            return []

    def export_metrics(self, file_path: str, hours: int = 24) -> bool:
        """
        导出指标数据到文件

        Args:
            file_path: 导出文件路径
            hours: 时间范围（小时）

        Returns:
            bool: 是否成功
        """
        try:
            export_data = {
                'export_time': datetime.now().isoformat(),
                'time_range_hours': hours,
                'system_metrics': self.get_historical_metrics('system', hours),
                'knowledge_base_metrics': self.get_historical_metrics('knowledge_base', hours),
                'performance_metrics': self.get_historical_metrics('performance', hours),
                'alert_history': [
                    {**alert, 'start_time': alert['start_time'].isoformat() if isinstance(alert['start_time'], datetime) else alert['start_time']}
                    for alert in self.alert_history
                ]
            }

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)

            self.logger.info(f"指标数据导出成功: {file_path}")
            return True

        except Exception as e:
            self.logger.error(f"导出指标数据失败: {str(e)}")
            return False


# 全局监控器实例
_system_monitor: Optional[SystemMonitor] = None
_monitor_lock = threading.Lock()


def get_system_monitor() -> SystemMonitor:
    """获取全局系统监控器实例"""
    global _system_monitor
    if _system_monitor is None:
        with _monitor_lock:
            if _system_monitor is None:
                _system_monitor = SystemMonitor()
    return _system_monitor


def cleanup_system_monitor():
    """清理全局系统监控器实例"""
    global _system_monitor
    if _system_monitor is not None:
        _system_monitor.stop_monitoring()
        _system_monitor = None