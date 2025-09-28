# 统一配置系统架构设计

## 设计原则

1. **分类管理**: 按功能模块分类配置
2. **层次化**: 支持默认配置、环境配置、用户配置
3. **类型安全**: 强类型验证，防止配置错误
4. **动态加载**: 支持运行时配置更新
5. **环境适配**: 支持开发、测试、生产环境
6. **向后兼容**: 保持现有API兼容性

## 配置分类体系

### 1. 系统配置 (system.yaml)
- 设备配置 (GPU/CPU)
- 路径配置 (模型、数据、日志)
- 环境变量管理

### 2. 模型配置 (models.yaml)
- 说话人分离模型配置
- ASR模型配置
- LLM模型配置
- Embedding模型配置

### 3. 处理配置 (processing.yaml)
- 音频处理参数
- 并发控制参数
- 批处理参数
- 超时和重试配置

### 4. 算法配置 (algorithms.yaml)
- 相似度阈值
- 清洗参数
- 压缩参数
- 质量控制参数

### 5. 业务配置 (business.yaml)
- 特定领域参数
- 客服对话相关配置
- 知识库配置

## 配置优先级

1. 命令行参数 (最高优先级)
2. 环境变量
3. 用户配置文件 (`config/local/*.yaml`)
4. 环境配置文件 (`config/environments/`)
5. 默认配置文件 (`config/defaults/`)

## 配置文件结构

```
config/
├── defaults/           # 默认配置
│   ├── system.yaml
│   ├── models.yaml
│   ├── processing.yaml
│   ├── algorithms.yaml
│   └── business.yaml
├── environments/       # 环境特定配置
│   ├── development.yaml
│   ├── testing.yaml
│   └── production.yaml
├── local/             # 本地覆盖配置 (git ignore)
│   └── *.yaml
├── schemas/           # 配置验证模式
│   └── config_schema.py
└── manager.py         # 配置管理器
```

## 配置加载机制

1. **初始化阶段**: 加载默认配置
2. **环境识别**: 根据环境变量加载环境配置
3. **本地覆盖**: 加载本地配置文件
4. **环境变量**: 覆盖特定配置项
5. **验证阶段**: 类型和约束验证
6. **缓存机制**: 配置缓存和热重载

## 配置访问接口

```python
from config import get_config, update_config

# 获取配置
device = get_config('system.device.cuda_device')
batch_size = get_config('processing.batch_size')

# 更新配置
update_config('processing.max_workers', 8)

# 获取整个配置组
model_config = get_config('models.asr')
```

## 向后兼容策略

- 保持现有模块构造函数API
- 提供配置迁移工具
- 逐步迁移，不破坏现有功能
- 提供配置验证和诊断工具