# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

这是一个端到端的客服对话录音转高质量知识库语料的系统。项目将客服对话wav录音文件转换为结构化的知识库语料，用于训练或改进AI客服系统。

## 核心技术栈

- **说话人分离**: `pyannote.audio` - 基于深度学习的说话人分离模型
- **语音识别**: `SenseVoice-Small` - 阿里开源的高精度多语言语音识别模型
- **音频处理**: `torchaudio` - PyTorch音频处理库
- **GPU加速**: CUDA支持，模型运行在GPU上

## 项目架构

### 数据流管道
```
原始录音(.wav/.mp3) → 说话人分离(pyannote) → 时间戳文件(.rttm) → 音频切分(torchaudio) → 语音识别(SenseVoice) → 知识库语料
```

### 核心组件

1. **说话人分离模块** (`download_pyannote.py`)
   - 使用 pyannote/speaker-diarization-3.1 模型
   - 输入: WAV音频文件
   - 输出: RTTM格式的说话人时间戳文件
   - GPU加速，支持进度监控

2. **音频切分模块** (待实现)
   - 根据RTTM文件切分原始音频
   - 按说话人和时间段生成子音频文件
   - 命名格式: `说话人-起始时间-结束时间.wav`

3. **语音识别模块** (SenseVoice-Small)
   - 本地部署的SenseVoice模型
   - 支持多语言识别和情感检测
   - 非自回归架构，推理速度快

## 目录结构

```
.
├── download_pyannote.py        # 主要处理脚本
├── requirement.md              # 项目需求文档
├── mp3s/                      # 原始MP3音频文件
├── wavs/                      # WAV格式音频文件
├── rttms/                     # 说话人分离结果文件
└── senseVoice-small/          # 本地SenseVoice模型
    ├── config.yaml           # 模型配置
    ├── configuration.json    # 模型参数
    └── README.md             # 模型说明文档
```

## 开发指南

### 环境要求
- Python 3.8+
- PyTorch (CUDA支持)
- CUDA兼容的GPU
- Hugging Face访问token (用于pyannote模型)
- 阿里云API密钥 (用于后续LLM清洗功能)

### 环境变量配置
创建 `.env` 文件并配置以下环境变量：
```bash
# Hugging Face 访问令牌 (用于pyannote模型)
export HUGGINGFACE_TOKEN=your_huggingface_token_here

# 阿里云API配置 (用于后续LLM清洗功能)
export DASHSCOPE_API_KEY=your_dashscope_api_key_here
export DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

参考 `.env.example` 文件获取完整的配置模板。

### 关键依赖
```python
# 说话人分离
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook

# 音频处理
import torchaudio
import torch

# 语音识别
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
```

### 开发原则
- 所有操作在 `download_pyannote.py` 中完成，避免代码冗余
- 优先考虑单个录音处理，后续扩展批量处理
- 保持GPU内存预加载以提升处理速度
- 遵循时间戳精确切分，不处理重叠问题

### RTTM格式说明
RTTM文件格式: `SPEAKER waveform 1 起始时间 持续时间 <NA> <NA> 说话人ID <NA> <NA>`
- 第3列: 起始时间(秒)
- 第4列: 持续时间(秒)
- 第7列: 说话人标识(SPEAKER_00, SPEAKER_01等)

### 音频切分规则
- 切分文件放置在 `wavs/原文件名/` 目录下
- 文件命名: `说话人ID-起始时间-结束时间.wav`
- 结束时间 = 起始时间 + 持续时间
- 严格按RTTM文件逐行切分，不处理时间重叠

### 当前实现状态
- ✅ 说话人分离 (pyannote.audio)
- ⏳ 音频切分 (需要实现)
- ⏳ 语音识别 (SenseVoice集成)
- ⏳ 批量处理和并发

### 注意事项
- 模型使用Hugging Face token，需要通过环境变量 `HUGGINGFACE_TOKEN` 配置
- pyannote模型自动加载到GPU (cuda:0)
- SenseVoice模型支持中文、英文、粤语、日语、韩语识别
- 所有处理过程保持音频质量，避免重采样损失
- 阿里云API配置用于后续LLM文本清洗功能