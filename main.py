
#!/usr/bin/env python3
"""
端到端音频处理系统
主入口文件
"""

import warnings
import os

# 过滤烦人的警告信息
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
warnings.filterwarnings("ignore", category=UserWarning, module="speechbrain")
warnings.filterwarnings("ignore", message=".*torchaudio._backend.*deprecated.*")
warnings.filterwarnings("ignore", message=".*TensorFloat-32.*")
warnings.filterwarnings("ignore", message=".*std().*degrees of freedom.*")

# 设置环境变量减少一些库的输出
os.environ['PYTHONWARNINGS'] = 'ignore'

from processor import AudioProcessor


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

# # 时间戳拼接算法，用以确保说话人语音的连续

# # senseVoice-small识别


# from modelscope.pipelines import pipeline
# from modelscope.utils.constant import Tasks

# inference_pipeline = pipeline(
#     task=Tasks.auto_speech_recognition,
#     model='/home/dzs-ai-4/dzs-dev/end2end_autio2kg/senseVoice-small',
#     model_revision="master",
#     device="cuda:0",)

# rec_result = inference_pipeline('/home/dzs-ai-4/dzs-dev/end2end_autio2kg/wavs/111.wav')
# print(rec_result)