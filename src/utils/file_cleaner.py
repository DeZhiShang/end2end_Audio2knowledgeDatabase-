"""
中间文件清理器
在问答对抽取完成后自动清理中间文件，节省磁盘空间
"""

import os
import shutil
from typing import Dict, Any, List, Optional
from pathlib import Path
from src.utils.logger import get_logger


class IntermediateFileCleaner:
    """中间文件清理器 - 安全清理处理完成的音频中间文件"""

    def __init__(self, enable_cleanup: bool = True, dry_run: bool = False):
        """
        初始化文件清理器

        Args:
            enable_cleanup: 是否启用自动清理
            dry_run: 是否为干运行模式（仅记录不实际删除）
        """
        self.logger = get_logger(__name__)
        self.enable_cleanup = enable_cleanup
        self.dry_run = dry_run

        # 基础路径配置，优先从配置系统获取
        try:
            from config import get_processing_paths, get_output_paths
            processing_paths = get_processing_paths()
            output_paths = get_output_paths()
            self.base_paths = {
                'rttm_dir': processing_paths['rttm_dir'],
                'wav_dir': processing_paths['wav_dir'],
                'docs_dir': output_paths['docs_dir']
            }
        except Exception:
            # 回退到硬编码默认值
            self.base_paths = {
                'rttm_dir': 'data/processed/rttms',
                'wav_dir': 'data/processed/wavs',
                'docs_dir': 'data/output/docs'
            }

        if not self.enable_cleanup:
            self.logger.info("中间文件清理已禁用")
        elif self.dry_run:
            self.logger.info("中间文件清理器启用 (干运行模式)")
        else:
            self.logger.info("中间文件清理器启用 (实际删除模式)")

    def extract_file_number(self, file_path: str) -> Optional[str]:
        """
        从文件路径中提取文件编号

        Args:
            file_path: 文件路径

        Returns:
            Optional[str]: 文件编号，如果无法提取则返回None
        """
        try:
            # 从路径中提取文件名（不含扩展名）
            file_name = Path(file_path).stem

            # 对于docs目录下的文件，直接使用文件名作为编号
            if self.base_paths['docs_dir'] in file_path:
                return file_name

            # 对于其他路径，也使用文件名作为编号
            return file_name

        except Exception as e:
            self.logger.warning(f"无法从路径中提取文件编号: {file_path}, 错误: {str(e)}")
            return None

    def get_intermediate_files(self, file_number: str) -> Dict[str, Any]:
        """
        根据文件编号获取所有相关的中间文件路径

        Args:
            file_number: 文件编号

        Returns:
            Dict[str, Any]: 包含各类中间文件路径的字典
        """
        intermediate_files = {
            'rttm_file': os.path.join(self.base_paths['rttm_dir'], f"{file_number}.rttm"),
            'wav_folder': os.path.join(self.base_paths['wav_dir'], file_number),
            'docs_file': os.path.join(self.base_paths['docs_dir'], f"{file_number}.md"),
            'exists': {}
        }

        # 检查文件是否存在
        intermediate_files['exists']['rttm'] = os.path.exists(intermediate_files['rttm_file'])
        intermediate_files['exists']['wav_folder'] = os.path.exists(intermediate_files['wav_folder'])
        intermediate_files['exists']['docs'] = os.path.exists(intermediate_files['docs_file'])

        return intermediate_files

    def calculate_disk_usage(self, paths: List[str]) -> float:
        """
        计算指定路径的磁盘使用量

        Args:
            paths: 文件或目录路径列表

        Returns:
            float: 磁盘使用量（MB）
        """
        total_size = 0

        for path in paths:
            try:
                if os.path.isfile(path):
                    total_size += os.path.getsize(path)
                elif os.path.isdir(path):
                    for dirpath, dirnames, filenames in os.walk(path):
                        for filename in filenames:
                            file_path = os.path.join(dirpath, filename)
                            try:
                                total_size += os.path.getsize(file_path)
                            except (OSError, FileNotFoundError):
                                pass  # 文件可能在遍历期间被删除
            except (OSError, FileNotFoundError):
                pass  # 路径不存在或无法访问

        return total_size / (1024 * 1024)  # 转换为MB

    def safe_remove_file(self, file_path: str) -> bool:
        """
        安全删除单个文件

        Args:
            file_path: 文件路径

        Returns:
            bool: 删除是否成功
        """
        try:
            if self.dry_run:
                self.logger.info(f"[DRY-RUN] 将删除文件: {file_path}")
                return True

            if os.path.exists(file_path):
                os.remove(file_path)
                self.logger.debug(f"已删除文件: {file_path}")
                return True
            else:
                self.logger.debug(f"文件不存在，跳过: {file_path}")
                return True

        except Exception as e:
            self.logger.error(f"删除文件失败: {file_path}, 错误: {str(e)}")
            return False

    def safe_remove_directory(self, dir_path: str) -> bool:
        """
        安全删除目录及其内容

        Args:
            dir_path: 目录路径

        Returns:
            bool: 删除是否成功
        """
        try:
            if self.dry_run:
                if os.path.exists(dir_path):
                    file_count = len([f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))])
                    self.logger.info(f"[DRY-RUN] 将删除目录: {dir_path} (包含 {file_count} 个文件)")
                return True

            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)
                self.logger.debug(f"已删除目录: {dir_path}")
                return True
            else:
                self.logger.debug(f"目录不存在，跳过: {dir_path}")
                return True

        except Exception as e:
            self.logger.error(f"删除目录失败: {dir_path}, 错误: {str(e)}")
            return False

    def cleanup_intermediate_files(self, file_path: str) -> Dict[str, Any]:
        """
        清理指定文件的所有中间文件

        Args:
            file_path: 触发清理的文件路径（通常是docs目录下的md文件）

        Returns:
            Dict[str, Any]: 清理结果统计
        """
        if not self.enable_cleanup:
            return {
                "success": False,
                "message": "中间文件清理已禁用",
                "cleaned_files": [],
                "disk_space_freed": 0.0
            }

        # 提取文件编号
        file_number = self.extract_file_number(file_path)
        if not file_number:
            return {
                "success": False,
                "error": f"无法从路径中提取文件编号: {file_path}",
                "cleaned_files": [],
                "disk_space_freed": 0.0
            }

        self.logger.info(f"开始清理文件编号: {file_number} 的中间文件...")

        # 获取中间文件信息
        intermediate_files = self.get_intermediate_files(file_number)

        # 计算清理前的磁盘使用量
        files_to_clean = []
        if intermediate_files['exists']['rttm']:
            files_to_clean.append(intermediate_files['rttm_file'])
        if intermediate_files['exists']['wav_folder']:
            files_to_clean.append(intermediate_files['wav_folder'])
        if intermediate_files['exists']['docs']:
            files_to_clean.append(intermediate_files['docs_file'])

        if not files_to_clean:
            self.logger.info(f"文件编号 {file_number} 没有找到需要清理的中间文件")
            return {
                "success": True,
                "message": "没有需要清理的中间文件",
                "file_number": file_number,
                "cleaned_files": [],
                "disk_space_freed": 0.0
            }

        # 计算清理前的大小
        disk_usage_before = self.calculate_disk_usage(files_to_clean)

        # 执行清理
        cleaned_files = []
        cleanup_results = []

        # 清理RTTM文件
        if intermediate_files['exists']['rttm']:
            success = self.safe_remove_file(intermediate_files['rttm_file'])
            cleanup_results.append(success)
            if success:
                cleaned_files.append({
                    'type': 'rttm',
                    'path': intermediate_files['rttm_file']
                })

        # 清理WAV文件夹
        if intermediate_files['exists']['wav_folder']:
            success = self.safe_remove_directory(intermediate_files['wav_folder'])
            cleanup_results.append(success)
            if success:
                cleaned_files.append({
                    'type': 'wav_folder',
                    'path': intermediate_files['wav_folder']
                })

        # 清理MD文件
        if intermediate_files['exists']['docs']:
            success = self.safe_remove_file(intermediate_files['docs_file'])
            cleanup_results.append(success)
            if success:
                cleaned_files.append({
                    'type': 'docs',
                    'path': intermediate_files['docs_file']
                })

        # 计算释放的磁盘空间
        disk_space_freed = disk_usage_before if all(cleanup_results) else 0.0

        # 统计结果
        total_operations = len(cleanup_results)
        successful_operations = sum(cleanup_results)

        success = successful_operations == total_operations and total_operations > 0

        if success:
            action_desc = "DRY-RUN清理" if self.dry_run else "清理"
            self.logger.info(f"✅ {action_desc}完成: {file_number}, 释放磁盘空间: {disk_space_freed:.2f}MB")
        else:
            self.logger.warning(f"⚠️ 清理部分成功: {file_number}, {successful_operations}/{total_operations} 个操作成功")

        return {
            "success": success,
            "file_number": file_number,
            "cleaned_files": cleaned_files,
            "disk_space_freed": disk_space_freed,
            "operations_total": total_operations,
            "operations_successful": successful_operations,
            "dry_run": self.dry_run
        }

    def get_cleanup_statistics(self) -> Dict[str, Any]:
        """
        获取清理器统计信息

        Returns:
            Dict[str, Any]: 统计信息
        """
        return {
            "enabled": self.enable_cleanup,
            "dry_run": self.dry_run,
            "base_paths": self.base_paths.copy()
        }


# 全局文件清理器实例
_file_cleaner: Optional[IntermediateFileCleaner] = None


def get_file_cleaner(enable_cleanup: bool = True, dry_run: bool = False) -> IntermediateFileCleaner:
    """
    获取文件清理器实例

    注意：不使用全局实例，每次调用都会根据参数创建新的实例，
    确保配置参数的正确性

    Args:
        enable_cleanup: 是否启用清理
        dry_run: 是否为干运行模式

    Returns:
        IntermediateFileCleaner: 文件清理器实例
    """
    return IntermediateFileCleaner(enable_cleanup=enable_cleanup, dry_run=dry_run)