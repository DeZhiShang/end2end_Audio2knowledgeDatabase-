"""
统一配置系统

提供简洁的配置访问接口，支持分层配置、环境特定配置和运行时配置更新。

Usage:
    from config import get_config, get_device, get_model_path

    # 获取具体配置值
    device = get_config('system.device.cuda_device')
    batch_size = get_config('processing.batch_processing.batch_size')

    # 使用便捷函数
    device = get_device()
    model_path = get_model_path('sensevoice')

    # 获取配置区块
    llm_config = get_config_section('models.llm')

    # 运行时更新配置
    update_config('processing.max_workers', 8)
"""

from .manager import (
    ConfigManager,
    get_config_manager,
    get_config,
    get_config_section,
    update_config,
    reload_config,
    get_device,
    get_model_path,
    get_api_config,
)

from .schemas.config_schema import (
    AppConfig,
    SystemConfig,
    ModelConfig,
    ProcessingConfig,
    AlgorithmConfig,
    BusinessConfig,
    DeviceConfig,
    PathConfig,
    LoggingConfig,
    MonitoringConfig,
    Environment,
    DeviceType,
    LogLevel,
)

__all__ = [
    # 核心管理器
    'ConfigManager',
    'get_config_manager',

    # 基础访问函数
    'get_config',
    'get_config_section',
    'update_config',
    'reload_config',

    # 便捷访问函数
    'get_device',
    'get_model_path',
    'get_api_config',

    # 配置模式类
    'AppConfig',
    'SystemConfig',
    'ModelConfig',
    'ProcessingConfig',
    'AlgorithmConfig',
    'BusinessConfig',
    'DeviceConfig',
    'PathConfig',
    'LoggingConfig',
    'MonitoringConfig',

    # 枚举类型
    'Environment',
    'DeviceType',
    'LogLevel',
]

# 版本信息
__version__ = '1.0.0'
__author__ = 'Audio2KG Team'
__description__ = 'Unified configuration system for audio2kg project'

# 初始化配置管理器（延迟加载）
def init_config(environment: str = None, config_root: str = None):
    """
    初始化配置系统

    Args:
        environment: 强制指定环境（development/testing/production）
        config_root: 配置文件根目录
    """
    import os

    if environment:
        os.environ['APP_ENV'] = environment

    if config_root:
        os.environ['CONFIG_ROOT'] = config_root

    # 触发配置管理器初始化
    manager = get_config_manager()
    return manager

# 配置诊断函数
def diagnose_config():
    """诊断配置系统状态"""
    manager = get_config_manager()

    print("=== Configuration System Diagnosis ===")
    print(f"Environment: {manager.get_environment()}")
    print(f"Config Root: {manager._config_root}")
    print(f"Loaded Files: {len(manager.get_loaded_files())}")

    for file_path in manager.get_loaded_files():
        print(f"  - {file_path}")

    # 验证配置
    errors = manager.validate_current_config()
    if errors:
        print(f"\nValidation Errors ({len(errors)}):")
        for error in errors:
            print(f"  - {error}")
    else:
        print("\nConfiguration validation: PASSED")

    # 显示主要配置值
    print(f"\nKey Configuration Values:")
    print(f"  Device: {get_device()}")
    print(f"  SenseVoice Model: {get_model_path('sensevoice')}")
    print(f"  Temperature: {get_config('models.llm.temperature')}")
    print(f"  Enable Async LLM: {get_config('processing.async_llm.enable_async')}")
    print(f"  Max Concurrent LLM: {get_config('processing.async_llm.max_concurrent_llm')}")

    api_config = get_api_config()
    print(f"  API Model: {api_config.get('model_name', 'Not configured')}")
    print(f"  API Base: {api_config.get('api_base', 'Not configured')}")
    print(f"  API Key: {'Configured' if api_config.get('api_key') else 'Not configured'}")

# 配置示例生成器
def generate_local_config_example():
    """生成本地配置示例文件"""
    import os
    from pathlib import Path

    manager = get_config_manager()
    local_dir = manager._config_root / 'local'
    local_dir.mkdir(exist_ok=True)

    example_content = """# 本地配置示例
# 复制此文件并重命名为对应的配置文件名（如 system.yaml）
# 只需要包含你想要覆盖的配置项

# 系统配置示例 (system.yaml)
device:
  cuda_device: "cuda:0"  # 修改GPU设备
  memory_fraction: 0.7   # 修改GPU内存使用比例

logging:
  level: "DEBUG"         # 修改日志级别

# 处理配置示例 (processing.yaml)
batch_processing:
  max_workers: 6         # 修改最大工作进程数
  batch_size: 32         # 修改批处理大小

# 模型配置示例 (models.yaml)
llm:
  temperature: 0.05      # 修改LLM温度参数
  max_tokens: 2048       # 修改最大token数
"""

    example_file = local_dir / 'example.yaml'
    with open(example_file, 'w', encoding='utf-8') as f:
        f.write(example_content)

    print(f"Local configuration example generated: {example_file}")
    print("Copy and rename this file to override specific configurations.")

# 配置迁移工具
def migrate_legacy_config():
    """从旧的配置方式迁移到新的配置系统"""
    print("=== Configuration Migration Tool ===")
    print("This tool helps migrate from hardcoded configurations to the new config system.")
    print("Please check the generated migration report and update your code accordingly.")

    # 这里可以添加具体的迁移逻辑
    # 例如扫描代码中的硬编码配置，生成迁移建议等
    pass