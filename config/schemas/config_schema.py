"""
配置验证模式
定义配置项的类型、约束和验证规则
"""

from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, field
from enum import Enum
import os


class DeviceType(Enum):
    """设备类型枚举"""
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"


class LogLevel(Enum):
    """日志级别枚举"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class Environment(Enum):
    """环境类型枚举"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"


@dataclass
class DeviceConfig:
    """设备配置"""
    cuda_device: str = "cuda:0"
    fallback_device: str = "cpu"
    auto_detect: bool = True
    memory_fraction: float = 0.8

    def __post_init__(self):
        if not 0.1 <= self.memory_fraction <= 1.0:
            raise ValueError("memory_fraction must be between 0.1 and 1.0")


@dataclass
class PathConfig:
    """路径配置"""
    project_root: str = "."
    data_root: str = "data"
    input_dir: str = "data/input"
    mp3_dir: str = "data/input/mp3s"
    processed_dir: str = "data/processed"
    wav_dir: str = "data/processed/wavs"
    rttm_dir: str = "data/processed/rttms"
    output_dir: str = "data/output"
    docs_dir: str = "data/output/docs"
    knowledge_base_file: str = "data/output/knowledgeDatabase.md"
    models_root: str = "models"
    sensevoice_model: str = "models/senseVoice-small"
    logs_dir: str = "logs"
    temp_dir: str = "temp"

    # 路径模板（用于动态生成文件路径）
    templates: Dict[str, str] = field(default_factory=lambda: {
        'rttm_file': '{rttm_dir}/{filename}.rttm',
        'wav_output_dir': '{wav_dir}/{filename}',
        'docs_file': '{docs_dir}/{filename}.md',
        'audio_segment': '{output_dir}/{segment_count:03d}_{speaker_id}-{start_time:.3f}-{end_time:.3f}.wav'
    })

    def get_absolute_path(self, path: str, base_path: Optional[str] = None) -> str:
        """获取绝对路径"""
        if os.path.isabs(path):
            return path

        if base_path is None:
            base_path = self.project_root

        return os.path.abspath(os.path.join(base_path, path))


@dataclass
class LoggingConfig:
    """日志配置"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    max_file_size: str = "10MB"
    backup_count: int = 5

    def __post_init__(self):
        if self.level not in [level.value for level in LogLevel]:
            raise ValueError(f"Invalid log level: {self.level}")


@dataclass
class MonitoringConfig:
    """监控配置"""
    enable_monitoring: bool = True
    monitoring_interval: int = 30
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'cpu_percent': 85,
        'memory_percent': 85,
        'disk_usage_percent': 90,
        'processing_error_rate': 0.1
    })

    def __post_init__(self):
        if self.monitoring_interval < 1:
            raise ValueError("monitoring_interval must be positive")


@dataclass
class EndpointsConfig:
    """端点和网络配置"""
    embedding: Dict[str, Any] = field(default_factory=lambda: {
        'url': 'http://localhost:8001/v1/embeddings',
        'model_name': 'qwen3-embedding-8b',
        'timeout': 60,
        'standard_timeout': 30
    })
    network: Dict[str, Any] = field(default_factory=lambda: {
        'default_timeout': 30,
        'max_retries': 3,
        'file_lock_timeout': 15.0,
        'thread_join_timeout': 5.0,
        'monitor_thread_timeout': 30,
        'queue_get_timeout': 1.0
    })
    delays: Dict[str, float] = field(default_factory=lambda: {
        'queue_check': 0.1,
        'retry_delay': 0.5,
        'error_recovery': 1.0,
        'async_task_delay': 0.1,
        'concurrency_cleanup_delay': 0.1
    })

@dataclass
class CompactionConfig:
    """压缩和处理配置"""
    qa_compactor: Dict[str, Any] = field(default_factory=lambda: {
        'scheduler': {
            'min_qa_pairs_threshold': 50,
            'compression_ratio_threshold': 0.1,
            'min_inactive_buffer_size': 20,
            'max_time_since_last_compaction': 60
        }
    })
    knowledge_integration: Dict[str, Any] = field(default_factory=lambda: {
        'final_compression_threshold': 5,
        'default_similarity_threshold': 0.7,
        'qa_similarity_threshold': 0.75
    })
    audio: Dict[str, Any] = field(default_factory=lambda: {
        'default_sample_rate': 16000,
        'embedding_dimensions': 4096
    })

@dataclass
class ConcurrencyConfig:
    """并发和重试配置"""
    async_llm: Dict[str, Any] = field(default_factory=lambda: {
        'max_concurrent_tasks': 16,
        'max_retries': 2
    })
    embedding: Dict[str, Any] = field(default_factory=lambda: {
        'max_retries': 3,
        'max_workers': 4,
        'batch_size': 35
    })
    file_operations: Dict[str, Any] = field(default_factory=lambda: {
        'max_age_minutes': 30
    })

@dataclass
class TokenLimitsConfig:
    """Token限制配置"""
    llm_cleaning: Dict[str, Any] = field(default_factory=lambda: {
        'check_tokens': 1000,
        'clean_tokens': 4000
    })
    qa_processing: Dict[str, Any] = field(default_factory=lambda: {
        'extraction_tokens': 32768,
        'compaction_tokens': 32768
    })

@dataclass
class SimilarityConfig:
    """相似度配置"""
    thresholds: Dict[str, float] = field(default_factory=lambda: {
        'default_similarity': 0.75,
        'knowledge_integration': 0.7,
        'qa_compaction': 0.75,
        'final_compression': 0.7
    })

@dataclass
class CacheConfig:
    """缓存配置"""
    lru_cache: Dict[str, int] = field(default_factory=lambda: {
        'config_manager_maxsize': 128
    })

@dataclass
class FileFormatsConfig:
    """文件格式配置"""
    audio_extensions: List[str] = field(default_factory=lambda: ['.wav', '.mp3'])
    document_extensions: List[str] = field(default_factory=lambda: ['.md', '.txt'])
    data_extensions: List[str] = field(default_factory=lambda: ['.json', '.yaml', '.yml'])

    filename_patterns: Dict[str, str] = field(default_factory=lambda: {
        'audio_segment': '{segment_count:03d}_{speaker_id}-{start_time:.3f}-{end_time:.3f}.wav',
        'asr_output': '{base_name}{suffix}.md',
        'json_log': 'audio_processor_{date}.json'
    })

@dataclass
class SystemConfig:
    """系统配置"""
    device: DeviceConfig = field(default_factory=DeviceConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    environment: Dict[str, str] = field(default_factory=dict)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    security: Dict[str, Any] = field(default_factory=dict)
    endpoints: EndpointsConfig = field(default_factory=EndpointsConfig)
    compaction: CompactionConfig = field(default_factory=CompactionConfig)
    concurrency: ConcurrencyConfig = field(default_factory=ConcurrencyConfig)
    token_limits: TokenLimitsConfig = field(default_factory=TokenLimitsConfig)
    similarity: SimilarityConfig = field(default_factory=SimilarityConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    file_formats: FileFormatsConfig = field(default_factory=FileFormatsConfig)


@dataclass
class ModelConfig:
    """模型配置"""
    speaker_diarization: Dict[str, Any] = field(default_factory=dict)
    asr: Dict[str, Any] = field(default_factory=dict)
    llm: Dict[str, Any] = field(default_factory=dict)
    embedding: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessingConfig:
    """处理配置"""
    audio: Dict[str, Any] = field(default_factory=dict)
    batch_processing: Dict[str, Any] = field(default_factory=dict)
    async_llm: Dict[str, Any] = field(default_factory=dict)
    memory_management: Dict[str, Any] = field(default_factory=dict)
    progress_tracking: Dict[str, Any] = field(default_factory=dict)
    error_handling: Dict[str, Any] = field(default_factory=dict)
    performance: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AlgorithmConfig:
    """算法配置"""
    similarity: Dict[str, Any] = field(default_factory=dict)
    qa_extraction: Dict[str, Any] = field(default_factory=dict)
    knowledge_compression: Dict[str, Any] = field(default_factory=dict)
    data_cleaning: Dict[str, Any] = field(default_factory=dict)
    quality_control: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BusinessConfig:
    """业务配置"""
    domain: Dict[str, Any] = field(default_factory=dict)
    customer_service: Dict[str, Any] = field(default_factory=dict)
    knowledge_base: Dict[str, Any] = field(default_factory=dict)
    terminology: Dict[str, Any] = field(default_factory=dict)
    compliance: Dict[str, Any] = field(default_factory=dict)
    output_formatting: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AppConfig:
    """应用配置根对象"""
    system: SystemConfig = field(default_factory=SystemConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    algorithms: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    business: BusinessConfig = field(default_factory=BusinessConfig)

    # 配置元信息
    _environment: str = "development"
    _config_files_loaded: List[str] = field(default_factory=list)
    _last_modified: Optional[float] = None

    def validate(self) -> bool:
        """验证配置完整性"""
        try:
            # 验证必要的路径存在
            paths = self.system.paths
            required_paths = [
                paths.project_root,
                paths.data_root,
                paths.models_root
            ]

            for path in required_paths:
                abs_path = paths.get_absolute_path(path)
                if not os.path.exists(abs_path):
                    raise ValueError(f"Required path does not exist: {abs_path}")

            # 验证设备配置
            device_config = self.system.device
            if device_config.cuda_device.startswith("cuda:"):
                device_id = device_config.cuda_device.split(":")[1]
                if not device_id.isdigit():
                    raise ValueError(f"Invalid CUDA device: {device_config.cuda_device}")

            return True
        except Exception as e:
            raise ValueError(f"Configuration validation failed: {str(e)}")

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        result = {}
        for field_name, field_value in self.__dict__.items():
            if field_name.startswith('_'):
                continue
            if hasattr(field_value, '__dict__'):
                result[field_name] = self._dataclass_to_dict(field_value)
            else:
                result[field_name] = field_value
        return result

    def _dataclass_to_dict(self, obj) -> Dict[str, Any]:
        """递归转换dataclass为字典"""
        if hasattr(obj, '__dict__'):
            result = {}
            for key, value in obj.__dict__.items():
                if hasattr(value, '__dict__'):
                    result[key] = self._dataclass_to_dict(value)
                else:
                    result[key] = value
            return result
        else:
            return obj


def validate_config_dict(config_dict: Dict[str, Any]) -> List[str]:
    """验证配置字典，返回错误列表"""
    errors = []

    # 基本结构验证
    required_sections = ['system', 'models', 'processing', 'algorithms', 'business']
    for section in required_sections:
        if section not in config_dict:
            errors.append(f"Missing required section: {section}")

    # 具体配置验证
    if 'system' in config_dict:
        system_config = config_dict['system']

        # 设备配置验证
        if 'device' in system_config:
            device_config = system_config['device']
            if 'memory_fraction' in device_config:
                memory_fraction = device_config['memory_fraction']
                if not isinstance(memory_fraction, (int, float)) or not 0.1 <= memory_fraction <= 1.0:
                    errors.append("system.device.memory_fraction must be between 0.1 and 1.0")

    return errors