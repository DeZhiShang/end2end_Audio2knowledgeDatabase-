#!/usr/bin/env python3
"""
端到端音频处理系统
根目录主入口文件 - 重构后的项目结构
"""

import sys
import os
import warnings

# 将项目根目录添加到Python路径中
sys.path.insert(0, os.path.dirname(__file__))

# 过滤烦人的警告信息
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
warnings.filterwarnings("ignore", category=UserWarning, module="speechbrain")
warnings.filterwarnings("ignore", message=".*torchaudio._backend.*deprecated.*")
warnings.filterwarnings("ignore", message=".*TensorFloat-32.*")
warnings.filterwarnings("ignore", message=".*std().*degrees of freedom.*")

# 设置环境变量减少一些库的输出
os.environ['PYTHONWARNINGS'] = 'ignore'

from src.utils.processor import AudioProcessor


def main():
    """主程序入口"""
    print("🎵 端到端音频处理系统")
    print("流程: MP3音频 → WAV转换 → 说话人分离 → 切分子音频 → ASR语音识别 → Gleaning多轮清洗 → 高质量知识库语料")
    print("=" * 90)

    # 创建音频处理器
    processor = AudioProcessor()

    # 执行批量处理（包含MP3转WAV预处理）
    processor.process_batch()


if __name__ == "__main__":
    main()