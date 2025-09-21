"""
统一日志系统模块
提供规范化的日志输出功能，替换print语句
"""

import logging
import os
import sys
from datetime import datetime
from typing import Optional, Dict, Any
import json
from pathlib import Path


class ColoredFormatter(logging.Formatter):
    """彩色日志格式化器，为控制台输出添加颜色"""

    # ANSI颜色代码
    COLORS = {
        'DEBUG': '\033[36m',    # 青色
        'INFO': '\033[32m',     # 绿色
        'WARNING': '\033[33m',  # 黄色
        'ERROR': '\033[31m',    # 红色
        'CRITICAL': '\033[35m', # 紫色
        'RESET': '\033[0m'      # 重置
    }

    # Emoji图标
    ICONS = {
        'DEBUG': '🔍',
        'INFO': '✅',
        'WARNING': '⚠️',
        'ERROR': '❌',
        'CRITICAL': '🚨'
    }

    def format(self, record):
        # 获取颜色和图标
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        icon = self.ICONS.get(record.levelname, '📝')
        reset = self.COLORS['RESET']

        # 格式化时间
        timestamp = datetime.fromtimestamp(record.created).strftime('%H:%M:%S')

        # 格式化模块名（简化路径）
        module_name = record.name.split('.')[-1] if '.' in record.name else record.name

        # 构建彩色输出
        formatted_msg = f"{color}{icon} [{timestamp}] {record.levelname} {module_name}: {record.getMessage()}{reset}"

        return formatted_msg


class JSONFormatter(logging.Formatter):
    """JSON格式化器，用于结构化日志输出"""

    def format(self, record):
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'module': record.name,
            'message': record.getMessage(),
            'file': record.filename,
            'line': record.lineno,
            'function': record.funcName
        }

        # 添加额外字段（如果存在）
        if hasattr(record, 'extra_data'):
            log_entry.update(record.extra_data)

        return json.dumps(log_entry, ensure_ascii=False)


def log_with_data(logger: logging.Logger, level: str, message: str, extra_data: Dict[str, Any] = None):
    """
    记录带有额外数据的日志

    Args:
        logger: 日志器实例
        level: 日志级别
        message: 日志消息
        extra_data: 额外的结构化数据
    """
    if extra_data:
        # 创建带有额外数据的LogRecord
        record = logger.makeRecord(
            logger.name, getattr(logging, level.upper()),
            '', 0, message, (), None
        )
        record.extra_data = extra_data
        logger.handle(record)
    else:
        getattr(logger, level.lower())(message)


class EnhancedLogger:
    """增强的日志器，支持额外数据"""

    def __init__(self, logger: logging.Logger):
        self._logger = logger

    def debug(self, message: str, extra_data: Dict[str, Any] = None):
        log_with_data(self._logger, 'DEBUG', message, extra_data)

    def info(self, message: str, extra_data: Dict[str, Any] = None):
        log_with_data(self._logger, 'INFO', message, extra_data)

    def warning(self, message: str, extra_data: Dict[str, Any] = None):
        log_with_data(self._logger, 'WARNING', message, extra_data)

    def error(self, message: str, extra_data: Dict[str, Any] = None):
        log_with_data(self._logger, 'ERROR', message, extra_data)

    def critical(self, message: str, extra_data: Dict[str, Any] = None):
        log_with_data(self._logger, 'CRITICAL', message, extra_data)


class LoggerManager:
    """日志管理器：统一管理项目中的所有日志"""

    _instance = None
    _loggers: Dict[str, EnhancedLogger] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)

        # 设置根日志级别
        self.console_level = logging.INFO
        self.file_level = logging.DEBUG

        # 初始化根日志配置
        self._setup_root_logger()

    def _setup_root_logger(self):
        """设置根日志器"""
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)

        # 清除现有处理器
        root_logger.handlers.clear()

        # 添加控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.console_level)
        console_handler.setFormatter(ColoredFormatter())
        root_logger.addHandler(console_handler)

        # 添加文件处理器
        log_file = self.log_dir / f"audio_processor_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(self.file_level)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        ))
        root_logger.addHandler(file_handler)

        # 添加JSON日志处理器
        json_log_file = self.log_dir / f"audio_processor_{datetime.now().strftime('%Y%m%d')}.json"
        json_handler = logging.FileHandler(json_log_file, encoding='utf-8')
        json_handler.setLevel(self.file_level)
        json_handler.setFormatter(JSONFormatter())
        root_logger.addHandler(json_handler)

    def get_logger(self, name: str) -> EnhancedLogger:
        """获取指定名称的日志器"""
        if name not in self._loggers:
            standard_logger = logging.getLogger(name)
            self._loggers[name] = EnhancedLogger(standard_logger)
        return self._loggers[name]

    def set_console_level(self, level):
        """设置控制台日志级别"""
        self.console_level = level
        for handler in logging.getLogger().handlers:
            if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stdout:
                handler.setLevel(level)

    def set_file_level(self, level):
        """设置文件日志级别"""
        self.file_level = level
        for handler in logging.getLogger().handlers:
            if isinstance(handler, logging.FileHandler):
                handler.setLevel(level)


# 全局日志管理器实例
_logger_manager = LoggerManager()


def get_logger(name: Optional[str] = None) -> EnhancedLogger:
    """
    获取日志器的便捷函数

    Args:
        name: 日志器名称，如果为None则使用调用模块的名称

    Returns:
        配置好的日志器实例
    """
    if name is None:
        # 自动获取调用者的模块名
        import inspect
        frame = inspect.currentframe().f_back
        module = inspect.getmodule(frame)
        name = module.__name__ if module else 'unknown'

    return _logger_manager.get_logger(name)


def set_log_level(console_level=None, file_level=None):
    """
    设置日志级别

    Args:
        console_level: 控制台日志级别
        file_level: 文件日志级别
    """
    if console_level is not None:
        _logger_manager.set_console_level(console_level)
    if file_level is not None:
        _logger_manager.set_file_level(file_level)


# 便捷的模块级日志函数
def debug(message: str, extra_data: Dict[str, Any] = None):
    """记录DEBUG级别日志"""
    logger = get_logger()
    logger.debug(message, extra_data)


def info(message: str, extra_data: Dict[str, Any] = None):
    """记录INFO级别日志"""
    logger = get_logger()
    logger.info(message, extra_data)


def warning(message: str, extra_data: Dict[str, Any] = None):
    """记录WARNING级别日志"""
    logger = get_logger()
    logger.warning(message, extra_data)


def error(message: str, extra_data: Dict[str, Any] = None):
    """记录ERROR级别日志"""
    logger = get_logger()
    logger.error(message, extra_data)


def critical(message: str, extra_data: Dict[str, Any] = None):
    """记录CRITICAL级别日志"""
    logger = get_logger()
    logger.critical(message, extra_data)