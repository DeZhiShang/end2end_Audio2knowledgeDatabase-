"""
统一配置管理器
负责加载、合并、验证和提供配置访问接口
"""

import os
import yaml
import json
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
import logging
from copy import deepcopy
import threading
from functools import lru_cache

from .schemas.config_schema import AppConfig, validate_config_dict, Environment


class ConfigManager:
    """配置管理器单例类"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized'):
            return

        self._initialized = True
        self._config: Optional[AppConfig] = None
        self._config_dict: Dict[str, Any] = {}
        self._environment = self._detect_environment()
        self._config_root = self._find_config_root()
        self._logger = self._setup_logger()

        # 加载配置
        self._load_all_configs()

    def _detect_environment(self) -> str:
        """检测当前运行环境"""
        env = os.getenv('APP_ENV', os.getenv('ENVIRONMENT', 'development')).lower()

        if env in ['dev', 'develop', 'development']:
            return 'development'
        elif env in ['test', 'testing']:
            return 'testing'
        elif env in ['prod', 'production']:
            return 'production'
        else:
            return 'development'

    def _find_config_root(self) -> Path:
        """查找配置文件根目录"""
        # 从当前文件位置开始查找
        current_path = Path(__file__).parent
        if (current_path / 'defaults').exists():
            return current_path

        # 从项目根目录查找
        project_root = Path.cwd()
        config_path = project_root / 'config'
        if config_path.exists() and (config_path / 'defaults').exists():
            return config_path

        # 创建默认配置目录
        config_path.mkdir(exist_ok=True)
        return config_path

    def _setup_logger(self) -> logging.Logger:
        """设置日志器"""
        logger = logging.getLogger('ConfigManager')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def _load_yaml_file(self, file_path: Path) -> Dict[str, Any]:
        """加载YAML配置文件"""
        try:
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = yaml.safe_load(f)
                    return content if content is not None else {}
            return {}
        except Exception as e:
            self._logger.error(f"Error loading config file {file_path}: {str(e)}")
            return {}

    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """深度合并配置字典"""
        if not isinstance(base, dict) or not isinstance(override, dict):
            return override if override is not None else base

        result = deepcopy(base)

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = deepcopy(value)

        return result

    def _load_all_configs(self):
        """加载所有配置文件"""
        self._logger.info(f"Loading configurations for environment: {self._environment}")

        # 1. 加载默认配置
        defaults_dir = self._config_root / 'defaults'
        config_files = ['system.yaml', 'models.yaml', 'processing.yaml', 'algorithms.yaml', 'business.yaml']

        merged_config = {}
        loaded_files = []

        for config_file in config_files:
            file_path = defaults_dir / config_file
            config_data = self._load_yaml_file(file_path)
            if config_data:
                # 将文件名去掉扩展名作为顶级键
                section_name = config_file.replace('.yaml', '').replace('.yml', '')
                merged_config[section_name] = config_data
                loaded_files.append(str(file_path))
                self._logger.debug(f"Loaded default config: {config_file}")

        # 2. 加载环境特定配置
        env_file = self._config_root / 'environments' / f'{self._environment}.yaml'
        env_config = self._load_yaml_file(env_file)
        if env_config:
            merged_config = self._merge_configs(merged_config, env_config)
            loaded_files.append(str(env_file))
            self._logger.debug(f"Loaded environment config: {env_file.name}")

        # 3. 加载本地覆盖配置
        local_dir = self._config_root / 'local'
        if local_dir.exists():
            for config_file in config_files:
                local_file = local_dir / config_file
                local_config = self._load_yaml_file(local_file)
                if local_config:
                    section_name = config_file.replace('.yaml', '').replace('.yml', '')
                    if section_name in merged_config:
                        merged_config[section_name] = self._merge_configs(
                            merged_config[section_name], local_config
                        )
                    else:
                        merged_config[section_name] = local_config
                    loaded_files.append(str(local_file))
                    self._logger.debug(f"Loaded local config: {config_file}")

        # 4. 应用环境变量覆盖
        self._apply_env_overrides(merged_config)

        # 5. 验证配置
        validation_errors = validate_config_dict(merged_config)
        if validation_errors:
            self._logger.warning(f"Configuration validation warnings: {validation_errors}")

        # 6. 存储配置
        self._config_dict = merged_config
        try:
            self._config = self._dict_to_dataclass(merged_config)
            self._config._environment = self._environment
            self._config._config_files_loaded = loaded_files
            self._config.validate()  # 最终验证
        except Exception as e:
            self._logger.error(f"Failed to create config dataclass: {str(e)}")
            # 继续使用字典格式
            self._config = None

        self._logger.info(f"Configuration loaded successfully. Files: {len(loaded_files)}")

    def _apply_env_overrides(self, config: Dict[str, Any]):
        """应用环境变量覆盖"""
        # 定义环境变量映射
        env_mappings = {
            'HUGGINGFACE_TOKEN': ['system', 'environment', 'huggingface_token'],
            'DASHSCOPE_API_KEY': ['models', 'llm', 'api_key'],
            'DASHSCOPE_BASE_URL': ['models', 'llm', 'api_base'],
            'APP_LOG_LEVEL': ['system', 'logging', 'level'],
            'APP_CUDA_DEVICE': ['system', 'device', 'cuda_device'],
            'APP_MAX_WORKERS': ['processing', 'batch_processing', 'max_workers'],
            'APP_BATCH_SIZE': ['processing', 'batch_processing', 'batch_size'],
        }

        for env_var, config_path in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                # 导航到配置路径并设置值
                current = config
                for key in config_path[:-1]:
                    if key not in current:
                        current[key] = {}
                    current = current[key]

                # 类型转换
                final_key = config_path[-1]
                if env_var in ['APP_MAX_WORKERS', 'APP_BATCH_SIZE']:
                    try:
                        env_value = int(env_value)
                    except ValueError:
                        self._logger.warning(f"Invalid integer value for {env_var}: {env_value}")
                        continue

                current[final_key] = env_value
                self._logger.debug(f"Applied environment override: {env_var} -> {'.'.join(config_path)}")

    def _dict_to_dataclass(self, config_dict: Dict[str, Any]) -> AppConfig:
        """将配置字典转换为数据类"""
        # 简化实现，直接使用字典初始化
        # 在实际项目中，可能需要更复杂的类型转换逻辑
        try:
            return AppConfig(**config_dict)
        except Exception:
            # 如果直接转换失败，使用默认配置并逐个更新
            app_config = AppConfig()
            # 这里可以添加更详细的字段映射逻辑
            return app_config

    @lru_cache(maxsize=128)  # 从配置文件控制
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        获取配置值

        Args:
            key_path: 配置键路径，使用点号分隔，例如 'system.device.cuda_device'
            default: 默认值

        Returns:
            配置值
        """
        keys = key_path.split('.')
        current = self._config_dict

        try:
            for key in keys:
                if isinstance(current, dict) and key in current:
                    current = current[key]
                else:
                    return default
            return current
        except Exception:
            return default

    def set(self, key_path: str, value: Any) -> bool:
        """
        设置配置值（运行时临时修改）

        Args:
            key_path: 配置键路径
            value: 新值

        Returns:
            是否成功设置
        """
        try:
            keys = key_path.split('.')
            current = self._config_dict

            # 导航到父级
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]

            # 设置最终值
            current[keys[-1]] = value

            # 清除缓存
            self.get.cache_clear()

            self._logger.debug(f"Configuration updated: {key_path} = {value}")
            return True
        except Exception as e:
            self._logger.error(f"Failed to set configuration {key_path}: {str(e)}")
            return False

    def get_section(self, section: str) -> Dict[str, Any]:
        """获取配置区块"""
        return self._config_dict.get(section, {})

    def reload(self) -> bool:
        """重新加载配置"""
        try:
            self.get.cache_clear()
            self._load_all_configs()
            self._logger.info("Configuration reloaded successfully")
            return True
        except Exception as e:
            self._logger.error(f"Failed to reload configuration: {str(e)}")
            return False

    def export_config(self, file_path: str, format: str = 'yaml') -> bool:
        """导出当前配置到文件"""
        try:
            export_path = Path(file_path)
            export_path.parent.mkdir(parents=True, exist_ok=True)

            if format.lower() == 'yaml':
                with open(export_path, 'w', encoding='utf-8') as f:
                    yaml.dump(self._config_dict, f, default_flow_style=False, allow_unicode=True)
            elif format.lower() == 'json':
                with open(export_path, 'w', encoding='utf-8') as f:
                    json.dump(self._config_dict, f, indent=2, ensure_ascii=False)
            else:
                raise ValueError(f"Unsupported export format: {format}")

            self._logger.info(f"Configuration exported to: {export_path}")
            return True
        except Exception as e:
            self._logger.error(f"Failed to export configuration: {str(e)}")
            return False

    def get_environment(self) -> str:
        """获取当前环境"""
        return self._environment

    def get_loaded_files(self) -> List[str]:
        """获取已加载的配置文件列表"""
        if self._config:
            return self._config._config_files_loaded
        return []

    def validate_current_config(self) -> List[str]:
        """验证当前配置"""
        return validate_config_dict(self._config_dict)

    @property
    def config(self) -> AppConfig:
        """获取配置对象"""
        return self._config

    @property
    def config_dict(self) -> Dict[str, Any]:
        """获取配置字典"""
        return deepcopy(self._config_dict)


# 全局配置管理器实例
_config_manager = None


def get_config_manager() -> ConfigManager:
    """获取配置管理器实例"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def get_config(key_path: str, default: Any = None) -> Any:
    """便捷函数：获取配置值"""
    return get_config_manager().get(key_path, default)


def get_config_section(section: str) -> Dict[str, Any]:
    """便捷函数：获取配置区块"""
    return get_config_manager().get_section(section)


def update_config(key_path: str, value: Any) -> bool:
    """便捷函数：更新配置值"""
    return get_config_manager().set(key_path, value)


def reload_config() -> bool:
    """便捷函数：重新加载配置"""
    return get_config_manager().reload()


# 便捷访问函数
def get_device() -> str:
    """获取设备配置"""
    return get_config('system.device.cuda_device', 'cuda:1')


def get_model_path(model_name: str) -> str:
    """获取模型路径"""
    if model_name == 'sensevoice':
        return get_config('system.paths.sensevoice_model', '/home/dzs-ai-4/dzs-dev/end2end_autio2kg/models/senseVoice-small')
    else:
        # 对于其他模型，返回None或抛出异常
        return None

def get_path_from_template(template_name: str, **kwargs) -> str:
    """根据路径模板生成文件路径"""
    # 获取路径模板
    template = get_config(f'system.paths.templates.{template_name}')
    if not template:
        raise ValueError(f"未找到路径模板: {template_name}")

    # 获取基础路径配置
    paths_config = get_config('system.paths', {})

    # 合并基础路径和传入的参数
    format_args = {**paths_config, **kwargs}

    try:
        return template.format(**format_args)
    except KeyError as e:
        raise ValueError(f"路径模板 '{template_name}' 缺少参数: {e}")

def format_file_paths(filename: str) -> dict:
    """为给定文件名生成标准的文件路径集合"""
    try:
        return {
            'rttm_file': get_path_from_template('rttm_file', filename=filename),
            'wav_output_dir': get_path_from_template('wav_output_dir', filename=filename),
            'docs_file': get_path_from_template('docs_file', filename=filename)
        }
    except Exception as e:
        # 如果模板系统失败，回退到硬编码路径以保持向后兼容
        return {
            'rttm_file': f"data/processed/rttms/{filename}.rttm",
            'wav_output_dir': f"data/processed/wavs/{filename}",
            'docs_file': f"data/output/docs/{filename}.md"
        }


def get_input_paths() -> Dict[str, str]:
    """获取输入相关的路径配置"""
    try:
        return {
            'input_dir': get_config('system.paths.input_dir', 'data/input'),
            'mp3_dir': get_config('system.paths.mp3_dir', 'data/input/mp3s'),
        }
    except Exception:
        return {
            'input_dir': 'data/input',
            'mp3_dir': 'data/input/mp3s',
        }

def get_processing_paths() -> Dict[str, str]:
    """获取处理过程中的路径配置"""
    try:
        return {
            'processed_dir': get_config('system.paths.processed_dir', 'data/processed'),
            'wav_dir': get_config('system.paths.wav_dir', 'data/processed/wavs'),
            'rttm_dir': get_config('system.paths.rttm_dir', 'data/processed/rttms'),
        }
    except Exception:
        return {
            'processed_dir': 'data/processed',
            'wav_dir': 'data/processed/wavs',
            'rttm_dir': 'data/processed/rttms',
        }

def get_output_paths() -> Dict[str, str]:
    """获取输出相关的路径配置"""
    try:
        return {
            'output_dir': get_config('system.paths.output_dir', 'data/output'),
            'docs_dir': get_config('system.paths.docs_dir', 'data/output/docs'),
        }
    except Exception:
        return {
            'output_dir': 'data/output',
            'docs_dir': 'data/output/docs',
        }

def ensure_directories(paths: Dict[str, str]) -> None:
    """确保指定的目录存在"""
    import os
    for path_name, path_value in paths.items():
        if path_value and not os.path.exists(path_value):
            try:
                os.makedirs(path_value, exist_ok=True)
            except Exception as e:
                logger.warning(f"无法创建目录 {path_value}: {e}")

def get_api_config() -> Dict[str, str]:
    """获取API配置"""
    return {
        'api_key': os.getenv('DASHSCOPE_API_KEY'),
        'api_base': os.getenv('DASHSCOPE_BASE_URL'),
    }