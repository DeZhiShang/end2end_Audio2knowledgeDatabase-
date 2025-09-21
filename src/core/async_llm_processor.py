"""
异步LLM处理器
提供异步的LLM清洗任务队列和并发控制
"""

import asyncio
import threading
from queue import Queue, Empty
from typing import Dict, Any, Optional, Callable
from concurrent.futures import ThreadPoolExecutor
import time
import os
from src.core.llm_cleaner import LLMDataCleaner
from src.utils.logger import get_logger


class AsyncLLMProcessor:
    """异步LLM处理器：管理LLM清洗任务的异步执行"""

    def __init__(self, max_concurrent_tasks: int = 4, max_retries: int = 2):
        """
        初始化异步LLM处理器

        Args:
            max_concurrent_tasks: 最大并发LLM任务数
            max_retries: 最大重试次数
        """
        self.logger = get_logger(__name__)
        self.max_concurrent_tasks = max_concurrent_tasks
        self.max_retries = max_retries

        # 任务队列和状态管理
        self.task_queue = Queue()
        self.active_tasks = {}  # task_id -> future
        self.completed_tasks = {}  # task_id -> result
        self.failed_tasks = {}  # task_id -> error

        # 并发控制
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_tasks)
        self.running = False
        self.worker_thread = None

        # 延迟初始化LLM清洗器
        self.llm_cleaner = None

        # 统计信息
        self.stats = {
            'total_submitted': 0,
            'total_completed': 0,
            'total_failed': 0,
            'total_processing_time': 0.0
        }

    def _initialize_llm_cleaner(self):
        """延迟初始化LLM清洗器"""
        if self.llm_cleaner is None:
            try:
                self.llm_cleaner = LLMDataCleaner()
                self.logger.info("异步LLM清洗器初始化成功")
            except Exception as e:
                self.logger.error(f"LLM清洗器初始化失败: {str(e)}")
                self.llm_cleaner = False
                raise

    def start(self):
        """启动异步处理器"""
        if self.running:
            self.logger.warning("异步LLM处理器已经在运行")
            return

        self.running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        self.logger.info(f"异步LLM处理器已启动 (最大并发: {self.max_concurrent_tasks})")

    def stop(self, wait_for_completion: bool = True):
        """
        停止异步处理器

        Args:
            wait_for_completion: 是否等待当前任务完成
        """
        self.running = False

        if wait_for_completion:
            # 等待队列中的任务完成
            self.wait_for_all_tasks()

        # 关闭线程池
        self.executor.shutdown(wait=wait_for_completion)

        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5.0)

        self.logger.info("异步LLM处理器已停止")

    def submit_task(self,
                   asr_file: str,
                   enable_gleaning: bool = True,
                   max_rounds: int = None,
                   quality_threshold: float = None,
                   priority: int = 0,
                   callback: Optional[Callable] = None) -> str:
        """
        提交LLM清洗任务到队列

        Args:
            asr_file: ASR结果文件路径
            enable_gleaning: 是否启用gleaning多轮清洗
            max_rounds: 最大清洗轮数
            quality_threshold: 质量阈值
            priority: 任务优先级（数字越大优先级越高）
            callback: 任务完成回调函数

        Returns:
            str: 任务ID
        """
        task_id = f"task_{int(time.time() * 1000)}_{self.stats['total_submitted']}"

        task = {
            'task_id': task_id,
            'asr_file': asr_file,
            'enable_gleaning': enable_gleaning,
            'max_rounds': max_rounds,
            'quality_threshold': quality_threshold,
            'priority': priority,
            'callback': callback,
            'submitted_at': time.time(),
            'retries': 0
        }

        self.task_queue.put(task)
        self.stats['total_submitted'] += 1

        self.logger.info(f"提交LLM清洗任务: {task_id} (文件: {os.path.basename(asr_file)})")
        return task_id

    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        获取任务状态

        Args:
            task_id: 任务ID

        Returns:
            Dict[str, Any]: 任务状态信息
        """
        if task_id in self.completed_tasks:
            return {
                'status': 'completed',
                'result': self.completed_tasks[task_id]
            }
        elif task_id in self.failed_tasks:
            return {
                'status': 'failed',
                'error': self.failed_tasks[task_id]
            }
        elif task_id in self.active_tasks:
            return {
                'status': 'processing'
            }
        else:
            # 检查是否在队列中
            for task in list(self.task_queue.queue):
                if task['task_id'] == task_id:
                    return {
                        'status': 'queued',
                        'queue_position': list(self.task_queue.queue).index(task)
                    }
            return {
                'status': 'not_found'
            }

    def wait_for_task(self, task_id: str, timeout: float = None) -> Dict[str, Any]:
        """
        等待特定任务完成

        Args:
            task_id: 任务ID
            timeout: 超时时间（秒）

        Returns:
            Dict[str, Any]: 任务结果
        """
        start_time = time.time()

        while True:
            status = self.get_task_status(task_id)

            if status['status'] in ['completed', 'failed', 'not_found']:
                return status

            if timeout and (time.time() - start_time) > timeout:
                return {
                    'status': 'timeout',
                    'error': f'等待任务{task_id}超时'
                }

            time.sleep(0.1)

    def wait_for_all_tasks(self, timeout: float = None) -> Dict[str, Any]:
        """
        等待所有任务完成

        Args:
            timeout: 超时时间（秒）

        Returns:
            Dict[str, Any]: 等待结果统计
        """
        start_time = time.time()

        while True:
            # 清理已完成的任务
            self._cleanup_completed_tasks()

            # 检查是否还有任务在处理
            queue_empty = self.task_queue.empty()
            no_active_tasks = len(self.active_tasks) == 0

            if queue_empty and no_active_tasks:
                break

            if timeout and (time.time() - start_time) > timeout:
                return {
                    'status': 'timeout',
                    'remaining_queue': self.task_queue.qsize(),
                    'active_tasks': len(self.active_tasks)
                }
            time.sleep(0.5)

        return {
            'status': 'completed',
            'total_completed': len(self.completed_tasks),
            'total_failed': len(self.failed_tasks)
        }

    def get_statistics(self) -> Dict[str, Any]:
        """获取处理统计信息"""
        # 先清理已完成的任务以获得准确统计
        self._cleanup_completed_tasks()

        return {
            **self.stats,
            'queue_size': self.task_queue.qsize(),
            'active_tasks': len(self.active_tasks),
            'completed_tasks': len(self.completed_tasks),
            'failed_tasks': len(self.failed_tasks),
            'average_processing_time': (
                self.stats['total_processing_time'] / max(1, self.stats['total_completed'])
            )
        }

    def _worker_loop(self):
        """工作线程主循环"""
        self.logger.info("异步LLM处理器工作线程已启动")

        while self.running:
            try:
                # 获取任务（非阻塞）
                try:
                    task = self.task_queue.get(timeout=1.0)
                except Empty:
                    continue

                # 先清理已完成的任务
                self._cleanup_completed_tasks()

                # 检查是否有空闲的worker
                if len(self.active_tasks) >= self.max_concurrent_tasks:
                    # 放回队列，稍后重试
                    self.task_queue.put(task)
                    time.sleep(0.1)
                    continue

                # 提交任务到线程池
                future = self.executor.submit(self._process_task, task)
                self.active_tasks[task['task_id']] = future

            except Exception as e:
                self.logger.error(f"工作线程异常: {str(e)}")
                time.sleep(1.0)

        self.logger.info("异步LLM处理器工作线程已退出")

    def _cleanup_completed_tasks(self):
        """清理已完成的任务"""
        completed_task_ids = []

        for task_id, future in list(self.active_tasks.items()):
            if future.done():
                completed_task_ids.append(task_id)

                try:
                    result = future.result()

                    # 检查是否为重试中的任务（这种情况不应该出现了）
                    if result.get('retry'):
                        # 重试任务不算完成，继续等待重新提交
                        self.logger.warning(f"发现重试任务结果: {task_id}")
                        continue

                    self.completed_tasks[task_id] = result
                    self.stats['total_completed'] += 1

                    # 调用回调函数
                    if 'callback' in result and result['callback']:
                        try:
                            result['callback'](result)
                        except Exception as e:
                            self.logger.warning(f"任务回调函数执行失败: {str(e)}")

                except Exception as e:
                    self.failed_tasks[task_id] = str(e)
                    self.stats['total_failed'] += 1
                    self.logger.error(f"任务{task_id}执行失败: {str(e)}")

        # 从活跃任务中移除已完成的任务
        for task_id in completed_task_ids:
            if task_id in self.active_tasks:
                del self.active_tasks[task_id]

    def _process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理单个LLM清洗任务

        Args:
            task: 任务信息

        Returns:
            Dict[str, Any]: 处理结果
        """
        task_id = task['task_id']
        asr_file = task['asr_file']
        start_time = time.time()

        try:
            # 初始化LLM清洗器
            self._initialize_llm_cleaner()

            if not self.llm_cleaner or self.llm_cleaner is False:
                raise Exception("LLM清洗器不可用")

            self.logger.info(f"开始处理任务 {task_id}: {os.path.basename(asr_file)}")

            # 执行LLM清洗
            result = self.llm_cleaner.clean_markdown_file(
                input_file=asr_file,
                output_file=asr_file,  # 覆盖原文件
                enable_gleaning=task['enable_gleaning'],
                max_rounds=task['max_rounds'],
                quality_threshold=task['quality_threshold']
            )

            processing_time = time.time() - start_time
            self.stats['total_processing_time'] += processing_time

            # 添加任务元信息
            result.update({
                'task_id': task_id,
                'processing_time': processing_time,
                'callback': task.get('callback')
            })

            if result['success']:
                self.logger.info(f"任务 {task_id} 处理完成 (耗时: {processing_time:.1f}s)")
            else:
                self.logger.warning(f"任务 {task_id} 处理失败: {result.get('error', '未知错误')}")

            return result

        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = str(e)

            # 重试逻辑
            if task['retries'] < self.max_retries:
                task['retries'] += 1
                self.logger.warning(f"任务 {task_id} 失败，准备重试 ({task['retries']}/{self.max_retries}): {error_msg}")

                # 延迟后重新提交
                time.sleep(min(2 ** task['retries'], 10))  # 指数退避
                self.task_queue.put(task)

                # 不返回结果，让线程继续执行
                raise Exception(f"重试中 ({task['retries']}/{self.max_retries}): {error_msg}")
            else:
                self.logger.error(f"任务 {task_id} 最终失败 (重试{self.max_retries}次): {error_msg}")
                return {
                    'task_id': task_id,
                    'success': False,
                    'error': error_msg,
                    'processing_time': processing_time,
                    'final_failure': True
                }


# 全局异步LLM处理器实例
_global_async_processor = None

def get_async_llm_processor() -> AsyncLLMProcessor:
    """获取全局异步LLM处理器实例"""
    global _global_async_processor
    if _global_async_processor is None:
        _global_async_processor = AsyncLLMProcessor()
    return _global_async_processor

def shutdown_async_llm_processor():
    """关闭全局异步LLM处理器"""
    global _global_async_processor
    if _global_async_processor:
        _global_async_processor.stop()
        _global_async_processor = None