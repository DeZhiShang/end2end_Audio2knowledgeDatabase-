"""
并发控制和线程安全机制
提供文件锁、原子操作和线程安全工具
"""

import os
import threading
import time
import fcntl
import contextlib
from typing import Dict, Any, Optional, Callable, TypeVar, List
from pathlib import Path
from datetime import datetime, timedelta
import uuid
import json
from dataclasses import dataclass

from src.utils.logger import get_logger

T = TypeVar('T')


@dataclass
class LockInfo:
    """锁信息"""
    lock_id: str
    thread_id: str
    process_id: int
    acquired_time: datetime
    lock_type: str
    resource: str


class FileLockManager:
    """文件锁管理器 - 提供跨进程文件锁功能"""

    def __init__(self):
        self.logger = get_logger(__name__)
        self._active_locks: Dict[str, LockInfo] = {}
        self._lock_registry_lock = threading.RLock()

    @contextlib.contextmanager
    def file_lock(self, file_path: str, mode: str = 'w', timeout: float = 30.0, exclusive: bool = True):
        """
        文件锁上下文管理器

        Args:
            file_path: 文件路径
            mode: 文件打开模式
            timeout: 超时时间（秒）
            exclusive: 是否独占锁

        Yields:
            file: 已锁定的文件对象
        """
        lock_id = str(uuid.uuid4())
        file_obj = None
        acquired = False

        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            # 打开文件
            file_obj = open(file_path, mode)

            # 尝试获取文件锁
            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    if exclusive:
                        fcntl.flock(file_obj.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    else:
                        fcntl.flock(file_obj.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)

                    acquired = True

                    # 记录锁信息
                    with self._lock_registry_lock:
                        self._active_locks[lock_id] = LockInfo(
                            lock_id=lock_id,
                            thread_id=str(threading.get_ident()),
                            process_id=os.getpid(),
                            acquired_time=datetime.now(),
                            lock_type="exclusive" if exclusive else "shared",
                            resource=file_path
                        )

                    self.logger.debug(f"获取文件锁成功: {file_path} ({lock_id})")
                    break

                except (IOError, OSError):
                    # 锁被占用，等待一小段时间后重试
                    time.sleep(0.1)

            if not acquired:
                raise TimeoutError(f"获取文件锁超时: {file_path}")

            yield file_obj

        finally:
            # 释放锁
            if acquired and file_obj:
                try:
                    fcntl.flock(file_obj.fileno(), fcntl.LOCK_UN)
                    with self._lock_registry_lock:
                        self._active_locks.pop(lock_id, None)
                    self.logger.debug(f"释放文件锁: {file_path} ({lock_id})")
                except Exception as e:
                    self.logger.error(f"释放文件锁失败: {str(e)}")

            # 关闭文件
            if file_obj:
                try:
                    file_obj.close()
                except Exception as e:
                    self.logger.error(f"关闭文件失败: {str(e)}")

    def get_active_locks(self) -> List[LockInfo]:
        """获取当前活跃的锁信息"""
        with self._lock_registry_lock:
            return list(self._active_locks.values())

    def cleanup_stale_locks(self, max_age_minutes: int = 30):
        """清理过期的锁记录"""
        cutoff_time = datetime.now() - timedelta(minutes=max_age_minutes)
        with self._lock_registry_lock:
            stale_locks = [
                lock_id for lock_id, lock_info in self._active_locks.items()
                if lock_info.acquired_time < cutoff_time
            ]
            for lock_id in stale_locks:
                self._active_locks.pop(lock_id, None)
            if stale_locks:
                self.logger.info(f"清理过期锁记录: {len(stale_locks)} 个")


class ThreadSafeCounter:
    """线程安全计数器"""

    def __init__(self, initial_value: int = 0):
        self._value = initial_value
        self._lock = threading.RLock()

    def increment(self, amount: int = 1) -> int:
        """递增计数器"""
        with self._lock:
            self._value += amount
            return self._value

    def decrement(self, amount: int = 1) -> int:
        """递减计数器"""
        with self._lock:
            self._value -= amount
            return self._value

    def get_value(self) -> int:
        """获取当前值"""
        with self._lock:
            return self._value

    def reset(self, value: int = 0) -> int:
        """重置计数器"""
        with self._lock:
            old_value = self._value
            self._value = value
            return old_value


class ThreadSafeDict:
    """线程安全字典"""

    def __init__(self):
        self._data: Dict[Any, Any] = {}
        self._lock = threading.RLock()

    def get(self, key: Any, default: Any = None) -> Any:
        """获取值"""
        with self._lock:
            return self._data.get(key, default)

    def set(self, key: Any, value: Any) -> None:
        """设置值"""
        with self._lock:
            self._data[key] = value

    def delete(self, key: Any) -> bool:
        """删除键值对"""
        with self._lock:
            if key in self._data:
                del self._data[key]
                return True
            return False

    def update(self, other: Dict[Any, Any]) -> None:
        """批量更新"""
        with self._lock:
            self._data.update(other)

    def keys(self) -> List[Any]:
        """获取所有键"""
        with self._lock:
            return list(self._data.keys())

    def values(self) -> List[Any]:
        """获取所有值"""
        with self._lock:
            return list(self._data.values())

    def items(self) -> List[tuple]:
        """获取所有键值对"""
        with self._lock:
            return list(self._data.items())

    def clear(self) -> None:
        """清空字典"""
        with self._lock:
            self._data.clear()

    def __len__(self) -> int:
        """获取长度"""
        with self._lock:
            return len(self._data)

    def __contains__(self, key: Any) -> bool:
        """检查键是否存在"""
        with self._lock:
            return key in self._data


class AtomicFileWriter:
    """原子文件写入器 - 确保文件写入的原子性"""

    def __init__(self, file_lock_manager: FileLockManager):
        self.logger = get_logger(__name__)
        self.file_lock_manager = file_lock_manager

    def write_json_atomic(self, file_path: str, data: Dict[str, Any], indent: int = 2) -> bool:
        """
        原子性写入JSON文件

        Args:
            file_path: 文件路径
            data: 要写入的数据
            indent: JSON缩进

        Returns:
            bool: 是否成功
        """
        try:
            temp_file = f"{file_path}.tmp.{uuid.uuid4()}"

            # 使用临时文件写入
            with self.file_lock_manager.file_lock(temp_file, 'w') as f:
                json.dump(data, f, ensure_ascii=False, indent=indent)
                f.flush()
                os.fsync(f.fileno())  # 强制刷新到磁盘

            # 原子性替换
            os.rename(temp_file, file_path)

            self.logger.debug(f"原子写入JSON文件成功: {file_path}")
            return True

        except Exception as e:
            self.logger.error(f"原子写入JSON文件失败: {file_path}, 错误: {str(e)}")
            # 清理临时文件
            try:
                if 'temp_file' in locals() and os.path.exists(temp_file):
                    os.remove(temp_file)
            except:
                pass
            return False

    def write_text_atomic(self, file_path: str, content: str, encoding: str = 'utf-8') -> bool:
        """
        原子性写入文本文件

        Args:
            file_path: 文件路径
            content: 文本内容
            encoding: 编码格式

        Returns:
            bool: 是否成功
        """
        try:
            temp_file = f"{file_path}.tmp.{uuid.uuid4()}"

            # 使用临时文件写入
            with self.file_lock_manager.file_lock(temp_file, 'w') as f:
                f.write(content)
                f.flush()
                os.fsync(f.fileno())  # 强制刷新到磁盘

            # 原子性替换
            os.rename(temp_file, file_path)

            self.logger.debug(f"原子写入文本文件成功: {file_path}")
            return True

        except Exception as e:
            self.logger.error(f"原子写入文本文件失败: {file_path}, 错误: {str(e)}")
            # 清理临时文件
            try:
                if 'temp_file' in locals() and os.path.exists(temp_file):
                    os.remove(temp_file)
            except:
                pass
            return False

    def read_json_safe(self, file_path: str, default: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        安全读取JSON文件

        Args:
            file_path: 文件路径
            default: 默认值

        Returns:
            Dict[str, Any]: 读取的数据
        """
        if default is None:
            default = {}

        if not os.path.exists(file_path):
            return default

        try:
            with self.file_lock_manager.file_lock(file_path, 'r', exclusive=False) as f:
                return json.load(f)

        except Exception as e:
            self.logger.error(f"安全读取JSON文件失败: {file_path}, 错误: {str(e)}")
            return default


class ConcurrencyMonitor:
    """并发监控器 - 监控并发操作状态"""

    def __init__(self):
        self.logger = get_logger(__name__)
        self._operations = ThreadSafeDict()
        self._counters = {
            'active_operations': ThreadSafeCounter(),
            'total_operations': ThreadSafeCounter(),
            'failed_operations': ThreadSafeCounter(),
            'lock_conflicts': ThreadSafeCounter()
        }

    @contextlib.contextmanager
    def monitor_operation(self, operation_name: str, operation_id: str = None):
        """
        监控操作执行

        Args:
            operation_name: 操作名称
            operation_id: 操作ID（可选）
        """
        if operation_id is None:
            operation_id = str(uuid.uuid4())

        start_time = datetime.now()
        operation_info = {
            'id': operation_id,
            'name': operation_name,
            'start_time': start_time,
            'thread_id': threading.get_ident(),
            'process_id': os.getpid()
        }

        # 记录操作开始
        self._operations.set(operation_id, operation_info)
        self._counters['active_operations'].increment()
        self._counters['total_operations'].increment()

        try:
            self.logger.debug(f"开始监控操作: {operation_name} ({operation_id})")
            yield operation_id

        except Exception as e:
            self._counters['failed_operations'].increment()
            self.logger.error(f"操作失败: {operation_name} ({operation_id}), 错误: {str(e)}")
            raise

        finally:
            # 记录操作完成
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            operation_info.update({
                'end_time': end_time,
                'duration': duration,
                'status': 'completed'
            })

            self._operations.delete(operation_id)
            self._counters['active_operations'].decrement()

            self.logger.debug(f"操作完成: {operation_name} ({operation_id}), 耗时: {duration:.2f}s")

    def get_active_operations(self) -> List[Dict[str, Any]]:
        """获取当前活跃的操作"""
        return [op for op in self._operations.values()]

    def get_statistics(self) -> Dict[str, Any]:
        """获取并发统计信息"""
        return {
            'active_operations': self._counters['active_operations'].get_value(),
            'total_operations': self._counters['total_operations'].get_value(),
            'failed_operations': self._counters['failed_operations'].get_value(),
            'lock_conflicts': self._counters['lock_conflicts'].get_value(),
            'active_operation_details': self.get_active_operations()
        }


class ResourcePool:
    """资源池 - 管理有限资源的并发访问"""

    def __init__(self, max_resources: int, resource_factory: Callable[[], T]):
        """
        初始化资源池

        Args:
            max_resources: 最大资源数量
            resource_factory: 资源创建工厂函数
        """
        self.logger = get_logger(__name__)
        self.max_resources = max_resources
        self.resource_factory = resource_factory

        self._available_resources: List[T] = []
        self._in_use_resources: Dict[str, T] = {}
        self._lock = threading.RLock()
        self._resource_available = threading.Condition(self._lock)

        # 预创建资源
        self._create_initial_resources()

    def _create_initial_resources(self):
        """预创建初始资源"""
        with self._lock:
            for _ in range(self.max_resources):
                try:
                    resource = self.resource_factory()
                    self._available_resources.append(resource)
                except Exception as e:
                    self.logger.error(f"创建资源失败: {str(e)}")

    @contextlib.contextmanager
    def acquire_resource(self, timeout: float = 30.0):
        """
        获取资源

        Args:
            timeout: 超时时间（秒）

        Yields:
            T: 资源对象
        """
        resource_id = str(uuid.uuid4())
        resource = None

        try:
            # 获取资源
            with self._resource_available:
                start_time = time.time()

                while not self._available_resources and (time.time() - start_time) < timeout:
                    remaining_time = timeout - (time.time() - start_time)
                    if remaining_time <= 0:
                        break
                    self._resource_available.wait(remaining_time)

                if not self._available_resources:
                    raise TimeoutError("获取资源超时")

                resource = self._available_resources.pop()
                self._in_use_resources[resource_id] = resource

            self.logger.debug(f"获取资源成功: {resource_id}")
            yield resource

        finally:
            # 释放资源
            if resource is not None:
                with self._lock:
                    self._in_use_resources.pop(resource_id, None)
                    self._available_resources.append(resource)
                    self._resource_available.notify()

                self.logger.debug(f"释放资源: {resource_id}")

    def get_pool_status(self) -> Dict[str, int]:
        """获取资源池状态"""
        with self._lock:
            return {
                'total_resources': self.max_resources,
                'available_resources': len(self._available_resources),
                'in_use_resources': len(self._in_use_resources)
            }


# 全局并发控制组件实例
_file_lock_manager: Optional[FileLockManager] = None
_atomic_writer: Optional[AtomicFileWriter] = None
_concurrency_monitor: Optional[ConcurrencyMonitor] = None
_component_lock = threading.Lock()


def get_file_lock_manager() -> FileLockManager:
    """获取全局文件锁管理器"""
    global _file_lock_manager
    if _file_lock_manager is None:
        with _component_lock:
            if _file_lock_manager is None:
                _file_lock_manager = FileLockManager()
    return _file_lock_manager


def get_atomic_writer() -> AtomicFileWriter:
    """获取全局原子文件写入器"""
    global _atomic_writer
    if _atomic_writer is None:
        with _component_lock:
            if _atomic_writer is None:
                _atomic_writer = AtomicFileWriter(get_file_lock_manager())
    return _atomic_writer


def get_concurrency_monitor() -> ConcurrencyMonitor:
    """获取全局并发监控器"""
    global _concurrency_monitor
    if _concurrency_monitor is None:
        with _component_lock:
            if _concurrency_monitor is None:
                _concurrency_monitor = ConcurrencyMonitor()
    return _concurrency_monitor


# 便捷装饰器
def thread_safe_operation(operation_name: str):
    """线程安全操作装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            monitor = get_concurrency_monitor()
            with monitor.monitor_operation(operation_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def with_file_lock(file_path_arg: str = 'file_path', exclusive: bool = True, timeout: float = 30.0):
    """文件锁装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # 获取文件路径参数
            if isinstance(file_path_arg, str):
                file_path = kwargs.get(file_path_arg)
                if file_path is None and args:
                    # 尝试从位置参数中获取
                    try:
                        import inspect
                        sig = inspect.signature(func)
                        params = list(sig.parameters.keys())
                        if file_path_arg in params:
                            idx = params.index(file_path_arg)
                            if idx < len(args):
                                file_path = args[idx]
                    except:
                        pass
            else:
                file_path = file_path_arg

            if not file_path:
                raise ValueError("无法获取文件路径参数")

            lock_manager = get_file_lock_manager()
            with lock_manager.file_lock(file_path, exclusive=exclusive, timeout=timeout):
                return func(*args, **kwargs)
        return wrapper
    return decorator