# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

这是一个端到端的客服对话录音转高质量知识库语料的系统。项目将客服对话wav录音文件转换为结构化的知识库语料，用于训练或改进AI客服系统。

## 核心技术栈

- **说话人分离**: `pyannote.audio` - 基于深度学习的说话人分离模型
- **语音识别**: `SenseVoice-Small` - 阿里开源的高精度多语言语音识别模型
- **LLM数据清洗**: `qwen-plus-latest` - 阿里云通义千问大语言模型
- **音频处理**: `torchaudio` - PyTorch音频处理库
- **GPU加速**: CUDA支持，模型运行在GPU上

## 项目架构

### 数据流管道
```
原始录音(.wav/.mp3) → 说话人分离(pyannote) → 时间戳文件(.rttm) → 音频切分(torchaudio) → 语音识别(SenseVoice) → LLM数据清洗(qwen-plus) → 高质量知识库语料
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

3. **语音识别模块** (`asr.py`)
   - 本地部署的SenseVoice模型
   - 支持多语言识别和情感检测
   - 非自回归架构，推理速度快

4. **LLM数据清洗模块** (`llm_cleaner.py`)
   - 使用qwen-plus-latest模型清洗ASR结果
   - 专针对博邦方舟无创血糖仪客服对话优化
   - 自动修正识别错误、过滤噪音、还原真实对话

## 目录结构

```
.
├── main.py                     # 主程序入口
├── processor.py                # 音频处理器（统一流程管理）
├── diarization.py             # 说话人分离模块
├── audio_segmentation.py      # 音频切分模块
├── asr.py                     # 语音识别模块
├── llm_cleaner.py             # LLM数据清洗模块
├── audio_converter.py         # 音频格式转换模块
├── test_llm_cleaning.py       # LLM清洗功能测试脚本
├── requirement.md             # 项目需求文档
├── LLM_CLEANING_GUIDE.md      # LLM清洗功能使用指南
├── mp3s/                      # 原始MP3音频文件
├── wavs/                      # WAV格式音频文件
├── rttms/                     # 说话人分离结果文件
├── docs/                      # ASR识别结果文件
├── docs_cleaned/              # LLM清洗后的高质量语料
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

# LLM数据清洗
import openai
from dotenv import load_dotenv
```

### 开发原则
- 模块化设计，各功能独立开发和测试
- 支持单个文件和批量处理模式
- 智能跳过机制，避免重复处理
- 保持GPU内存预加载以提升处理速度
- 遵循时间戳精确切分，不处理重叠问题
- 容错设计，单个模块失败不影响整体流程

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
- ✅ 音频切分 (torchaudio)
- ✅ 语音识别 (SenseVoice-Small)
- ✅ LLM数据清洗 (qwen-plus-latest)
- ✅ 批量处理和智能跳过
- ✅ 端到端处理流程

### 注意事项
- 模型使用Hugging Face token，需要通过环境变量 `HUGGINGFACE_TOKEN` 配置
- pyannote模型自动加载到GPU (cuda:0)
- SenseVoice模型支持中文、英文、粤语、日语、韩语识别
- 所有处理过程保持音频质量，避免重采样损失
- 阿里云API配置用于LLM数据清洗功能，自动优化ASR识别结果

## 使用方法

### 快速开始
```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置环境变量
cp .env.example .env
# 编辑.env文件，填入API密钥

# 3. 运行主程序
python main.py
```

### 功能模块

1. **完整端到端处理**
   ```bash
   python main.py
   ```
   自动执行：MP3转WAV → 说话人分离 → 音频切分 → ASR识别 → LLM清洗

2. **仅LLM数据清洗**
   ```bash
   python test_llm_cleaning.py
   ```

3. **自定义配置**
   ```python
   from processor import AudioProcessor

   processor = AudioProcessor()
   # 禁用LLM清洗
   processor.process_batch(enable_llm_cleaning=False)
   # 强制重新处理
   processor.process_batch(force_overwrite=True)
   ```

### 输出结果
- **原始ASR结果**: `docs/filename.md`
- **清洗后语料**: `docs_cleaned/filename.md`
- **处理日志**: 控制台实时显示进度和统计信息