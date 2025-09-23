"""
知识库管理模块
实现Append-Only+Compact双缓存系统，管理问答对知识库
"""

import os
import json
import threading
import time
import uuid
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import fcntl
from pathlib import Path

from src.utils.logger import get_logger
from src.utils.concurrency import FileLockManager


class ProcessingStatus(Enum):
    """文件处理状态枚举"""
    PROCESSING = "processing"
    ASR_COMPLETED = "asr_completed"
    LLM_COMPLETED = "llm_completed"
    CLEAN_FINISHED = "clean_finished"
    QA_EXTRACTED = "qa_extracted"
    COMPACTED = "compacted"


@dataclass
class QAPair:
    """问答对数据结构"""
    id: str
    question: str
    answer: str
    source_file: str
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QAPair':
        """从字典创建实例"""
        data = data.copy()
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)

    def to_markdown(self) -> str:
        """转换为Markdown格式"""
        return f"""## Q: {self.question}

**A:** {self.answer}

---
"""


@dataclass
class FileStatus:
    """文件处理状态"""
    file_path: str
    status: ProcessingStatus
    last_updated: datetime
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'file_path': self.file_path,
            'status': self.status.value,
            'last_updated': self.last_updated.isoformat(),
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FileStatus':
        """从字典创建实例"""
        return cls(
            file_path=data['file_path'],
            status=ProcessingStatus(data['status']),
            last_updated=datetime.fromisoformat(data['last_updated']),
            metadata=data.get('metadata', {})
        )


class BufferSnapshot:
    """缓冲区快照"""
    def __init__(self, buffer_data: List[QAPair], offset: int):
        self.data = buffer_data.copy()
        self.offset = offset
        self.created_at = datetime.now()


class DualBufferKnowledgeBase:
    """
    双缓存知识库系统
    实现Append-Only + Compact架构，支持高并发写入和无冲突压缩
    """

    def __init__(self, knowledge_base_file: str = "data/output/knowledgeDatabase.md"):
        """
        初始化双缓存知识库

        Args:
            knowledge_base_file: 知识库主文件路径
        """
        self.logger = get_logger(__name__)
        self.knowledge_base_file = knowledge_base_file

        # 创建输出目录
        os.makedirs(os.path.dirname(knowledge_base_file), exist_ok=True)

        # 双缓冲区
        self.buffer_a: List[QAPair] = []
        self.buffer_b: List[QAPair] = []
        self.active_buffer = "A"  # 当前活跃缓冲区

        # 快照系统
        self.snapshot_offset = 0
        self.current_snapshot: Optional[BufferSnapshot] = None

        # 线程安全机制
        self.write_lock = threading.RLock()  # 写入锁
        self.compact_lock = threading.RLock()  # 压缩锁
        self.switch_lock = threading.RLock()  # 缓冲区切换锁
        self.file_lock_manager = FileLockManager()  # 文件锁管理器

        # 状态跟踪
        self.file_status_map: Dict[str, FileStatus] = {}
        self.status_file = os.path.join(os.path.dirname(knowledge_base_file), "processing_status.json")

        # 统计信息
        self.stats = {
            'total_qa_pairs': 0,
            'total_writes': 0,
            'total_compacts': 0,
            'last_compact_time': None,
            'active_buffer_size': 0,
            'inactive_buffer_size': 0
        }

        # 初始化
        self._load_existing_data()
        self._load_file_status()

        self.logger.info("知识库系统初始化完成")

    def _get_active_buffer(self) -> List[QAPair]:
        """获取当前活跃缓冲区"""
        return self.buffer_a if self.active_buffer == "A" else self.buffer_b

    def _get_inactive_buffer(self) -> List[QAPair]:
        """获取当前非活跃缓冲区"""
        return self.buffer_b if self.active_buffer == "A" else self.buffer_a

    def _load_existing_data(self):
        """加载现有的知识库数据"""
        if os.path.exists(self.knowledge_base_file):
            try:
                with open(self.knowledge_base_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                # 解析现有的问答对（简单实现，可根据实际格式调整）
                qa_pairs = self._parse_markdown_qa_pairs(content)
                self.buffer_a = qa_pairs
                self.stats['total_qa_pairs'] = len(qa_pairs)

                pass  # 静默加载知识库数据
            except Exception as e:
                self.logger.warning(f"加载现有知识库数据失败: {str(e)}")

    def _parse_markdown_qa_pairs(self, content: str) -> List[QAPair]:
        """解析Markdown格式的问答对（简化实现）"""
        # 这里是一个简化的解析实现，实际可以根据具体格式优化
        qa_pairs = []
        lines = content.split('\n')

        current_qa = None
        i = 0

        while i < len(lines):
            line = lines[i].strip()

            # 检测问题行
            if line.startswith('## Q:'):
                if current_qa:
                    qa_pairs.append(current_qa)

                question = line[5:].strip()

                # 查找答案行
                i += 1
                while i < len(lines) and not lines[i].strip().startswith('**A:**'):
                    i += 1

                if i < len(lines):
                    answer_line = lines[i].strip()
                    answer = answer_line[5:].strip() if answer_line.startswith('**A:**') else ""

                    # 创建问答对
                    current_qa = QAPair(
                        id=str(uuid.uuid4()),
                        question=question,
                        answer=answer,
                        source_file="existing_data",
                        timestamp=datetime.now()
                    )

            i += 1

        if current_qa:
            qa_pairs.append(current_qa)

        return qa_pairs

    def _load_file_status(self):
        """加载文件处理状态"""
        if os.path.exists(self.status_file):
            try:
                with open(self.status_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                for file_data in data.get('files', []):
                    file_status = FileStatus.from_dict(file_data)
                    self.file_status_map[file_status.file_path] = file_status

                pass  # 静默加载文件状态
            except Exception as e:
                self.logger.warning(f"加载文件状态失败: {str(e)}")

    def _save_file_status(self):
        """保存文件处理状态"""
        try:
            data = {
                'last_updated': datetime.now().isoformat(),
                'files': [status.to_dict() for status in self.file_status_map.values()]
            }

            with open(self.status_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

        except Exception as e:
            self.logger.error(f"保存文件状态失败: {str(e)}")

    def update_file_status(self, file_path: str, status: ProcessingStatus, metadata: Dict[str, Any] = None):
        """
        更新文件处理状态

        Args:
            file_path: 文件路径
            status: 处理状态
            metadata: 元数据
        """
        with self.write_lock:
            file_status = FileStatus(
                file_path=file_path,
                status=status,
                last_updated=datetime.now(),
                metadata=metadata or {}
            )

            self.file_status_map[file_path] = file_status
            self._save_file_status()

            pass  # 静默更新文件状态

    def get_file_status(self, file_path: str) -> Optional[FileStatus]:
        """
        获取文件处理状态

        Args:
            file_path: 文件路径

        Returns:
            FileStatus: 文件状态，如果不存在返回None
        """
        return self.file_status_map.get(file_path)

    def get_clean_finished_files(self) -> List[str]:
        """
        获取所有处理完成的文件列表

        Returns:
            List[str]: clean_finished状态的文件路径列表
        """
        return [
            file_path for file_path, status in self.file_status_map.items()
            if status.status == ProcessingStatus.CLEAN_FINISHED
        ]

    def append_qa_pairs(self, qa_pairs: List[QAPair], auto_save: bool = True) -> bool:
        """
        追加问答对到活跃缓冲区（线程安全）

        Args:
            qa_pairs: 问答对列表
            auto_save: 是否自动保存到文件（默认True，并发场景建议False）

        Returns:
            bool: 是否成功
        """
        if not qa_pairs:
            return True

        try:
            with self.write_lock:
                active_buffer = self._get_active_buffer()
                active_buffer.extend(qa_pairs)

                self.stats['total_writes'] += 1
                self.stats['active_buffer_size'] = len(active_buffer)
                self.stats['total_qa_pairs'] += len(qa_pairs)

                pass  # 静默追加问答对

                # 根据参数决定是否自动保存到文件
                if auto_save:
                    self._save_to_file()

                return True

        except Exception as e:
            self.logger.error(f"追加问答对失败: {str(e)}")
            return False

    def save(self) -> bool:
        """
        手动保存当前状态到文件

        Returns:
            bool: 是否保存成功
        """
        try:
            self._save_to_file()
            return True
        except Exception as e:
            self.logger.error(f"手动保存失败: {str(e)}")
            return False

    def _save_to_file(self):
        """保存当前状态到文件（文件锁保护）"""
        try:
            # 合并两个缓冲区的数据
            all_qa_pairs = self.buffer_a + self.buffer_b

            # 按时间戳排序
            all_qa_pairs.sort(key=lambda qa: qa.timestamp)

            # 生成Markdown内容
            content = self._generate_markdown_content(all_qa_pairs)

            # 使用互斥锁和原子性写入保护整个过程
            lock_file = f"{self.knowledge_base_file}.lock"
            temp_file = f"{self.knowledge_base_file}.tmp"

            # 获取文件锁保护整个写入过程
            with self.file_lock_manager.file_lock(lock_file, mode='w', timeout=15.0, exclusive=True):
                # 在锁保护下进行原子性写入
                try:
                    # 写入临时文件
                    with open(temp_file, 'w', encoding='utf-8') as f:
                        f.write(content)
                        f.flush()  # 确保数据写入磁盘
                        os.fsync(f.fileno())  # 强制同步到磁盘

                    # 原子性替换文件
                    os.rename(temp_file, self.knowledge_base_file)

                    self.logger.debug(f"知识库文件保存成功: {len(all_qa_pairs)} 个问答对")

                except Exception as inner_e:
                    # 清理临时文件
                    if os.path.exists(temp_file):
                        try:
                            os.remove(temp_file)
                        except:
                            pass
                    raise inner_e

        except Exception as e:
            self.logger.error(f"保存知识库文件失败: {str(e)}")
            # 最终清理
            temp_file = f"{self.knowledge_base_file}.tmp"
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass

    def _generate_markdown_content(self, qa_pairs: List[QAPair]) -> str:
        """生成Markdown格式的知识库内容"""
        header = f"""# 博邦方舟无创血糖仪知识库

> 最后更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
>
> 总计问答对: {len(qa_pairs)} 个
>
> 数据来源: 客服对话录音 → 说话人分离 → ASR识别 → LLM清洗 → 问答对抽取

---

"""

        content_parts = [header]

        for qa in qa_pairs:
            content_parts.append(qa.to_markdown())

        return "\n".join(content_parts)

    def create_snapshot(self) -> Optional[BufferSnapshot]:
        """
        为当前活跃缓冲区创建快照（用于压缩）

        Returns:
            BufferSnapshot: 快照对象，失败返回None
        """
        try:
            with self.compact_lock:
                active_buffer = self._get_active_buffer()
                snapshot = BufferSnapshot(active_buffer, len(active_buffer))

                self.current_snapshot = snapshot
                self.snapshot_offset = len(active_buffer)

                self.logger.info(f"创建缓冲区快照: {len(snapshot.data)} 个问答对 (偏移: {self.snapshot_offset})")

                return snapshot

        except Exception as e:
            self.logger.error(f"创建快照失败: {str(e)}")
            return None

    def switch_buffers_with_tail_sync(self, compacted_data: List[QAPair]) -> bool:
        """
        缓冲区切换并同步尾部增量数据

        Args:
            compacted_data: 压缩后的数据

        Returns:
            bool: 是否成功
        """
        try:
            with self.switch_lock:
                # 在切换前获取当前缓冲区引用
                current_active_buffer = self._get_active_buffer()
                current_inactive_buffer = self._get_inactive_buffer()

                # 获取快照点之后的增量数据（尾部数据）
                tail_data = current_active_buffer[self.snapshot_offset:] if self.snapshot_offset < len(current_active_buffer) else []

                # 准备新的活跃缓冲区数据：压缩数据 + 尾部增量数据
                new_active_data = compacted_data.copy()
                if tail_data:
                    new_active_data.extend(tail_data)
                    self.logger.info(f"同步尾部增量数据: {len(tail_data)} 个问答对")

                # 切换活跃缓冲区指针
                old_active = self.active_buffer
                self.active_buffer = "B" if self.active_buffer == "A" else "A"

                # 现在获取新的活跃和非活跃缓冲区引用
                new_active_buffer = self._get_active_buffer()
                new_inactive_buffer = self._get_inactive_buffer()

                # 设置新的活跃缓冲区数据（原来的非活跃缓冲区）
                new_active_buffer.clear()
                new_active_buffer.extend(new_active_data)

                # 清空新的非活跃缓冲区（原来的活跃缓冲区）
                new_inactive_buffer.clear()

                # 更新统计信息
                self.stats['active_buffer_size'] = len(new_active_buffer)
                self.stats['inactive_buffer_size'] = len(new_inactive_buffer)
                self.stats['total_compacts'] += 1
                self.stats['last_compact_time'] = datetime.now().isoformat()

                # 重置快照状态
                self.snapshot_offset = 0
                self.current_snapshot = None

                pass  # 静默切换缓冲区
                self.logger.info(f"新活跃缓冲区数据: {len(compacted_data)} 压缩 + {len(tail_data)} 尾部 = {len(new_active_buffer)} 总计")

                # 保存到文件
                self._save_to_file()

                return True

        except Exception as e:
            self.logger.error(f"缓冲区切换失败: {str(e)}")
            return False

    def get_all_qa_pairs(self) -> List[QAPair]:
        """
        获取所有问答对（只读操作）

        Returns:
            List[QAPair]: 所有问答对列表
        """
        with self.write_lock:
            all_pairs = self.buffer_a + self.buffer_b
            return sorted(all_pairs, key=lambda qa: qa.timestamp)

    def search_qa_pairs(self, query: str, limit: int = 10) -> List[QAPair]:
        """
        搜索问答对

        Args:
            query: 搜索关键词
            limit: 返回结果限制

        Returns:
            List[QAPair]: 匹配的问答对列表
        """
        all_pairs = self.get_all_qa_pairs()

        matching_pairs = []
        query_lower = query.lower()

        for qa in all_pairs:
            if (query_lower in qa.question.lower() or
                query_lower in qa.answer.lower()):
                matching_pairs.append(qa)

                if len(matching_pairs) >= limit:
                    break

        return matching_pairs

    def get_statistics(self) -> Dict[str, Any]:
        """
        获取知识库统计信息

        Returns:
            Dict[str, Any]: 统计信息
        """
        with self.write_lock:
            self.stats.update({
                'active_buffer_size': len(self._get_active_buffer()),
                'inactive_buffer_size': len(self._get_inactive_buffer()),
                'total_qa_pairs': len(self.buffer_a) + len(self.buffer_b),
                'current_active_buffer': self.active_buffer,
                'file_status_count': len(self.file_status_map),
                'clean_finished_files': len(self.get_clean_finished_files())
            })

            return self.stats.copy()

    def cleanup(self):
        """清理资源"""
        try:
            # 保存最终状态
            self._save_to_file()
            self._save_file_status()

            pass  # 静默资源清理

        except Exception as e:
            self.logger.error(f"知识库清理失败: {str(e)}")


# 全局知识库实例（单例模式）
_knowledge_base_instance: Optional[DualBufferKnowledgeBase] = None
_instance_lock = threading.Lock()


def get_knowledge_base() -> DualBufferKnowledgeBase:
    """
    获取全局知识库实例（单例模式）

    Returns:
        DualBufferKnowledgeBase: 知识库实例
    """
    global _knowledge_base_instance

    if _knowledge_base_instance is None:
        with _instance_lock:
            if _knowledge_base_instance is None:
                _knowledge_base_instance = DualBufferKnowledgeBase()

    return _knowledge_base_instance


def cleanup_knowledge_base():
    """清理全局知识库实例"""
    global _knowledge_base_instance

    if _knowledge_base_instance is not None:
        _knowledge_base_instance.cleanup()
        _knowledge_base_instance = None