"""
说话人分离模块
负责使用pyannote.audio进行说话人分离
"""

import warnings
# 过滤相关警告
warnings.filterwarnings("ignore", category=UserWarning)

from pyannote.audio import Pipeline
import torch
from pyannote.audio.pipelines.utils.hook import ProgressHook
import torchaudio


class SpeakerDiarization:
    """说话人分离处理器"""

    def __init__(self, model_name="pyannote/speaker-diarization-3.1",
                 auth_token="UNKNOWN",
                 device="cuda:0"):
        """
        初始化说话人分离模型

        Args:
            model_name: 模型名称
            auth_token: HuggingFace访问令牌
            device: 计算设备
        """
        self.pipeline = Pipeline.from_pretrained(
            model_name,
            use_auth_token=auth_token
        )
        self.pipeline.to(torch.device(device))
        print(f"说话人分离模型已加载到 {device}")

    def process(self, wav_file, num_speakers=2):
        """
        执行说话人分离

        Args:
            wav_file: 音频文件路径
            num_speakers: 说话人数量

        Returns:
            diarization: 分离结果对象
        """
        # 加载音频文件
        waveform, sample_rate = torchaudio.load(wav_file)

        # 执行说话人分离
        with ProgressHook() as hook:
            diarization = self.pipeline(
                {"waveform": waveform, "sample_rate": sample_rate},
                hook=hook,
                num_speakers=num_speakers
            )

        return diarization

    def save_rttm(self, diarization, rttm_file):
        """
        保存分离结果为RTTM文件

        Args:
            diarization: 分离结果对象
            rttm_file: RTTM文件路径
        """
        with open(rttm_file, "w") as rttm:
            diarization.write_rttm(rttm)