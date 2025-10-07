[English](README.md) | [简体中文](README.zh-CN.md)

# End-to-End Customer Service Audio Knowledge Base Extraction System

> 🎯 Enterprise-grade batch MP3 audio to high-quality knowledge base end-to-end solution

An audio processing system designed specifically for customer service scenarios that can batch convert large amounts of customer service dialogue recordings into structured high-quality knowledge base corpus, providing reliable data foundation for RAG intelligent customer service.

## ⚡ Core Highlights

### 🎯 Solving Real Business Pain Points
- **End-to-End Batch Processing**: One-click processing of large amounts of MP3 recording files without manual intervention
- **Enterprise-Grade Architecture**: Unified configuration management, asynchronous processing, intelligent resource management
- **High-Quality Output**: Ensuring knowledge base quality through multi-round cleaning and intelligent compression

### 🚀 Technical Innovation Features
- **Gleaning Multi-Round Cleaning Mechanism**: LLM-driven iterative optimization, significantly improving text quality
- **Asynchronous LLM Processor**: Supporting high-concurrency processing, dramatically improving processing efficiency
- **Dual-Cache Compression System**: Smart compression algorithms preventing unlimited knowledge base expansion
- **Catastrophic Forgetting Prevention**: Innovative incremental compression mechanism maintaining knowledge base consistency
- **Automatic Resource Management**: Dynamic cleanup of intermediate files ensuring controllable system resources

### 🎨 Processing Pipeline Advantages
Compared to traditional ASR solutions, this system specifically optimizes:
- **High-Noise Environments**: Complex acoustic environment processing of customer service recordings
- **Clear Role Identification**: Precise speaker separation and dialogue restoration
- **Complete Workflow**: End-to-end open-source solution filling market gaps

## 📋 System Architecture

### End-to-End Processing Pipeline

```
Original MP3 Audio
    ↓ [Audio Format Conversion]
WAV Standard Format Audio
    ↓ [Speaker Separation - pyannote-3.1]
RTTM Timestamp Files
    ↓ [Audio Segment Splitting - torchaudio]
Speaker-Separated Audio Segments
    ↓ [Speech Recognition - SenseVoice-Small]
Raw ASR Text
    ↓ [Multi-Round LLM Cleaning - qwen-plus]
High-Quality Dialogue Corpus
    ↓ [Q&A Pair Extraction]
Structured Q&A Pairs
    ↓ [Intelligent Compression System]
High-Quality Knowledge Base
```

### Core Technology Stack

#### 🧠 AI Model Layer
- **Speaker Separation**: `pyannote/speaker-diarization-3.1` - Deep learning-based precise speaker identification
- **Speech Recognition**: `SenseVoice-Small` - Alibaba open-source multilingual ASR model, locally deployed
- **Text Processing**: `qwen-plus-latest` - Alibaba Cloud Tongyi Qianwen large language model

#### 🏗️ System Architecture Layer
- **Configuration Management**: Layered YAML configuration system supporting multi-environment deployment
- **Concurrent Processing**: ThreadPoolExecutor-based asynchronous task scheduling
- **Audio Processing**: PyTorch ecosystem torchaudio audio processing library
- **Intelligent Compression**: LLM-driven similarity verification and knowledge merging

#### 📊 Data Processing Layer
- **Format Conversion**: Intelligent audio format detection and conversion
- **Precise Splitting**: Millisecond-level audio segmentation based on RTTM timestamps
- **Quality Control**: Multi-round Gleaning cleaning and quality assessment
- **Knowledge Construction**: Structured Q&A pair extraction and hierarchical organization

## 🛠️ Quick Start

### Environment Requirements

- **Python**: 3.12
- **GPU**: CUDA-compatible graphics card (GTX 1080+ recommended, theoretically pyannote+senseVoice requires ~2GB VRAM)
- **Memory**: 16GB+ RAM (32GB recommended)
- **Storage**: 50GB+ available space

### Installation Steps

1. **Clone the Project**
```bash
git clone <repository-url>
cd end2end_autio2kg
```

2. **Install Dependencies**
```bash
# Install PyTorch (CUDA version)
pip install torch torchvision torchaudio

# Install project dependencies
pip install -r requirements.txt
```

3. **Configure Environment Variables**
```bash
cp .env.example .env
# Edit the .env file to configure necessary API keys
```

Required configurations:
```bash
# Hugging Face access token (for pyannote model)
HUGGINGFACE_TOKEN=hf_your_token_here

# Alibaba Cloud API configuration (for LLM cleaning)
DASHSCOPE_API_KEY=sk-your_api_key_here
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

4. **Prepare Models**
```bash
# SenseVoice model will be automatically downloaded to models/senseVoice-small/ on first run
```

### Running the System

1. **Place Audio Files**
```bash
# Place MP3 files in the data/input/ directory
cp your_recordings/*.mp3 data/input/
```

2. **Start Processing**
```bash
python main.py
```

3. **View Results**
```bash
# Processing results will be saved in the data/output/ directory
# Final knowledge base file: data/output/knowledgeDatabase.md
```

### Example Demonstration

**Input Data Example**:
![Input Example](images/input_example.png)

**Output Data Example**:
![Output Example](images/ouput_example.png)

## 📁 Project Structure

```
end2end_autio2kg/
├── main.py                     # Main program entry
├── requirements.txt            # Project dependencies
├── .env.example               # Environment variable template
│
├── config/                    # 🔧 Unified configuration management system
│   ├── manager.py             # Configuration manager core
│   ├── schemas/               # Configuration schema definitions
│   ├── defaults/              # Default configuration files
│   │   ├── system.yaml        # System configuration
│   │   ├── models.yaml        # Model configuration
│   │   ├── processing.yaml    # Processing configuration
│   │   ├── compaction.yaml    # Compression configuration
│   │   ├── concurrency.yaml   # Concurrency configuration
│   │   ├── algorithms.yaml    # Algorithm configuration
│   │   └── business.yaml      # Business configuration
│   ├── environments/          # Environment-specific configuration
│   └── local/                 # Local override configuration
│
├── src/                       # 🧩 Core source code
│   ├── core/                  # Core processing modules
│   │   ├── diarization.py     # Speaker separation
│   │   ├── audio_segmentation.py # Audio segmentation
│   │   ├── asr.py             # Speech recognition
│   │   ├── llm_cleaner.py     # LLM data cleaning
│   │   ├── async_llm_processor.py # Asynchronous LLM processor
│   │   ├── qa_extractor.py    # Q&A extractor
│   │   ├── qa_compactor.py    # Q&A compactor
│   │   ├── knowledge_base.py  # Knowledge base management (dual-cache system)
│   │   ├── knowledge_integration.py # Knowledge base integration
│   │   └── embedding_similarity.py # Similarity calculation
│   └── utils/                 # Utility modules
│       ├── audio_converter.py # Audio format conversion
│       ├── processor.py       # Unified processor
│       ├── logger.py          # Log management
│       ├── concurrency.py     # Concurrency control
│       └── file_cleaner.py    # File cleanup tool
│
├── data/                      # 📊 Data directory
│   ├── input/                 # Input data (MP3 files)
│   ├── processed/             # Processing data
│   │   ├── wavs/             # WAV format files
│   │   └── rttms/            # Speaker separation results
│   └── output/               # Output data
│       └── docs/             # Final processing results
│
└── models/                   # 🤖 Model directory
    └── senseVoice-small/    # Local SenseVoice model
```

## ⚙️ Configuration System

### Layered Configuration Architecture

The system adopts layered configuration management, supporting flexible environment configuration and parameter tuning:

```yaml
# Configuration priority: Environment variables > Local config > Environment config > Default config
config/
├── defaults/           # Basic default configuration
├── environments/       # Environment-specific configuration (dev/test/prod)
└── local/             # Local override configuration (not version controlled)
```

### Core Configuration Description

#### System Configuration (`system.yaml`)
```yaml
device:
  cuda_device: "cuda:1"     # GPU device setting

paths:
  project_root: "."         # Project root path
  sensevoice_model: "models/senseVoice-small"  # ASR model path
```

#### Processing Configuration (`processing.yaml`)
```yaml
batch:
  enable_async_llm: true         # Enable asynchronous LLM processing
  enable_knowledge_base: true    # Enable knowledge base integration
  enable_gleaning: true          # Enable Gleaning multi-round cleaning
  max_gleaning_rounds: 3         # Maximum cleaning rounds
```

#### Concurrency Configuration (`concurrency.yaml`)
```yaml
async_llm:
  max_concurrent_tasks: 16       # Maximum concurrent LLM tasks
  max_retries: 2                 # Maximum retry count
```

## 🔄 Processing Pipeline Details

### 1. Audio Preprocessing Stage
- **Format Detection**: Automatically identify audio formats and convert to WAV uniformly
- **Quality Verification**: Check audio integrity and sampling rate
- **Path Management**: Standardize file naming and directory structure

### 2. Speaker Separation Stage
- **Model Loading**: pyannote/speaker-diarization-3.1 deep learning model
- **Separation Processing**: Generate RTTM format timestamp files
- **GPU Acceleration**: Support CUDA acceleration, significantly improving processing speed

### 3. Audio Segmentation Stage
- **Precise Segmentation**: Millisecond-level segmentation based on RTTM timestamps
- **File Naming**: `{sequence}_{speaker_ID}-{start_time}-{end_time}.wav`
- **Batch Processing**: Support parallel segmentation of large-scale audio files

### 4. Speech Recognition Stage
- **Local Deployment**: SenseVoice-Small model local inference
- **Multilingual Support**: Support Chinese-English mixed recognition
- **High-Precision Output**: Recognition accuracy optimized for customer service scenarios

### 5. LLM Cleaning Stage (Core Innovation)
- **Gleaning Mechanism**: Multi-round iterative cleaning, each round optimized based on previous results
- **Asynchronous Processing**: Support 16 concurrent tasks, dramatically improving processing efficiency
- **Quality Control**: Intelligent assessment of cleaning effects, dynamic adjustment of cleaning strategies
- **Professional Optimization**: Specialized cleaning strategies for medical device customer service dialogues

### 6. Q&A Extraction Stage
- **Intelligent Recognition**: LLM-based automatic Q&A pair identification and extraction
- **Structured Output**: Generate standard format Q&A pair data
- **Quality Assessment**: Quality scoring and filtering of extracted Q&A pairs

### 7. Intelligent Compression Stage (Core Innovation)
- **Similarity Verification**: LLM-based intelligent similarity judgment (93%+ confidence)
- **Layered Compression**: Embedding pre-filtering + LLM precise judgment
- **Forgetting Avoidance**: Incremental compression mechanism maintaining historical knowledge integrity
- **Dual-Cache System**: Active/inactive buffer design optimizing compression performance

## 🎯 Core Algorithms

### Gleaning Multi-Round Cleaning Algorithm

```python
# Multi-round iterative cleaning process
for round in range(max_rounds):
    cleaned_text = llm_clean(
        text=previous_result,
        context=business_context,
        round_number=round
    )

    quality_score = evaluate_quality(cleaned_text)
    if quality_score > threshold:
        break

    previous_result = cleaned_text
```

**Features**:
- Each round optimized based on previous results
- Dynamic quality assessment and early termination
- Specialized cleaning strategies for customer service dialogue scenarios

### Intelligent Compression Algorithm

```python
# Three-stage compression process
def compress_qa_pairs(qa_pairs):
    # Stage 1: Embedding pre-filtering
    candidates = embedding_prefilter(qa_pairs, threshold=0.85)

    # Stage 2: LLM precise similarity judgment
    similar_groups = llm_similarity_check(candidates)

    # Stage 3: Intelligent merging
    compressed_pairs = llm_merge_similar(similar_groups)

    return compressed_pairs
```

**Features**:
- 93%+ similarity detection confidence
- 40%+ compression rate improvement (compared to traditional algorithms)
- Avoiding catastrophic forgetting in dynamic knowledge base expansion

### Asynchronous Concurrency Control

```python
# Asynchronous LLM processor
class AsyncLLMProcessor:
    def __init__(self, max_concurrent=16):
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent)
        self.task_queue = Queue()

    async def process_batch(self, texts):
        tasks = [self.submit_task(text) for text in texts]
        results = await asyncio.gather(*tasks)
        return results
```

**Features**:
- Support for arbitrarily configured maximum concurrent LLM tasks
- Intelligent task scheduling and resource management
- Automatic error recovery and retry mechanisms

## 🔧 Advanced Configuration

### Environment Configuration Switching

```bash
# Development environment
export APP_ENV=development

# Testing environment
export APP_ENV=testing

# Production environment
export APP_ENV=production
```

### Performance Optimization Configuration

```yaml
# config/local/performance.yaml
system:
  device:
    cuda_device: "cuda:0"  # Specify GPU device

processing:
  batch:
    max_concurrent_tasks: 32  # Increase concurrency (high-end GPU)

algorithms:
  similarity:
    embedding_threshold: 0.90  # Increase pre-filtering threshold
```

### Business Scenario Customization

```yaml
# config/local/business.yaml
business:
  domain: "medical_device"      # Business domain
  terminology:                 # Professional terminology
    - "血糖仪"
    - "无创检测"
    - "校准"
```

## 🚀 Deployment Guide

### Production Environment Deployment

1. **Environment Configuration**
```bash
export APP_ENV=production
export APP_LOG_LEVEL=INFO
export APP_CUDA_DEVICE=cuda:0
```

2. **Resource Optimization**
```bash
# Ensure sufficient GPU memory
nvidia-smi

# Adjust system parameters
ulimit -n 65536  # Increase file descriptor limit
```

3. **Monitoring Configuration**
```bash
# Monitor GPU usage
nvidia-smi -l 1

# Monitor processing logs
tail -f logs/process.log
```

## 🔍 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```
Solutions:
- Reduce concurrent task count: max_concurrent_tasks: 8
- Use CPU mode: cuda_device: "cpu"
- Increase GPU memory or use higher-end graphics card
```

**2. API Call Failure**
```
Solutions:
- Check API key configuration in .env file
- Verify network connection and API service status
- Check detailed error logs in logs/ directory
```

**3. Model Download Failure**
```
Solutions:
- Ensure normal network connection
- Configure correct HUGGINGFACE_TOKEN
- Manually download model to models/ directory
```

**4. Unsupported Audio File Format**
```
Solutions:
- Ensure audio files are in MP3 or WAV format
- Check file integrity and encoding format
- Use ffmpeg to preprocess audio files
```

### Debug Mode

```bash
# Enable detailed logging
export APP_LOG_LEVEL=DEBUG

# View configuration diagnostics
python -c "from config import diagnose_config; diagnose_config()"

# Module testing
python -m src.core.asr          # Test ASR module
python -m src.core.diarization  # Test speaker separation
```

## 🤝 Contribution Guide

### Development Environment Setup

```bash
# 1. Fork the project and clone
git clone <your-fork-url>
cd end2end_autio2kg

# 2. Create development branch
git checkout -b feature/your-feature

# 3. Install development dependencies
pip install -r requirements.txt

# 4. Run tests
python -m pytest tests/  # If tests exist
```

### Code Standards

- **Configuration-Driven**: All hardcoded values should be managed through the configuration system
- **Type Safety**: Use dataclass and type annotations
- **Error Handling**: Implement graceful error handling and recovery mechanisms
- **Logging**: Use structured logging for key operations
- **Modular Design**: Maintain decoupling and independence between modules

### Commit Standards

```bash
# Commit message format
git commit -m "feat: Add custom ASR model support"
git commit -m "fix: Fix memory leak in concurrent processing"
git commit -m "docs: Update deployment guide documentation"
```

## 📄 License

This project adopts the [MIT License](LICENSE) open source protocol.

## 👥 Author Team

**DezSmart Medical Technology Co., Ltd. Technical Team**

- 📧 **Contact Email**: ericsenyao@163.com
- 🏢 **Company**: DezSmart Medical Technology Co., Ltd.

## 🙏 Acknowledgments

Thanks to the following open source projects and technical communities for their support:

- [pyannote.audio](https://github.com/pyannote/pyannote-audio) - Speaker separation technology
- [ModelScope](https://modelscope.cn/) - SenseVoice speech recognition model
- [Alibaba Cloud Tongyi Qianwen](https://dashscope.aliyuncs.com/) - LLM data processing service
- [PyTorch](https://pytorch.org/) - Deep learning framework

## 📞 Technical Support

For technical support or business cooperation, please contact us through:

- **Technical Issues**: Submit on GitHub Issues
- **Business Cooperation**: ericsenyao@163.com
- **Documentation Contribution**: Welcome to submit PRs for documentation improvements

---

<div align="center">

**⭐ If this project helps you, please give us a Star! ⭐**

</div>