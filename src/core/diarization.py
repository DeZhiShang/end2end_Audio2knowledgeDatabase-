"""
说话人分离模块
负责使用pyannote.audio进行说话人分离
"""

import warnings
import os
# 过滤相关警告
warnings.filterwarnings("ignore", category=UserWarning)

from pyannote.audio import Pipeline
import torch
import torchaudio
from src.utils.logger import get_logger

# 导入配置系统
from config import get_config


class SpeakerDiarization:
    """说话人分离处理器"""

    def __init__(self, model_name=None, auth_token=None, device=None, num_speakers=None):
        """
        初始化说话人分离模型

        Args:
            model_name: 模型名称，如果为None则从配置获取
            auth_token: HuggingFace访问令牌，如果为None则从配置获取
            device: 计算设备，如果为None则从配置获取
            num_speakers: 默认说话人数量，如果为None则从配置获取
        """
        self.logger = get_logger(__name__)

        # 从配置系统获取参数
        self.model_name = model_name or get_config('models.speaker_diarization.model_name', 'pyannote/speaker-diarization-3.1')
        self.device = device or get_config('models.speaker_diarization.device', 'cuda:1')
        self.num_speakers = num_speakers or get_config('models.speaker_diarization.num_speakers', 2)

        # 获取认证token
        if auth_token is None:
            auth_token = get_config('system.environment.huggingface_token') or os.getenv('HUGGINGFACE_TOKEN')
            if auth_token is None:
                self.logger.warning("未配置HUGGINGFACE_TOKEN，可能无法下载某些模型")

        # 初始化模型
        try:
            self.pipeline = Pipeline.from_pretrained(
                self.model_name,
                # use_auth_token=auth_token  # 如果需要token，取消注释
            )
            self.pipeline.to(torch.device(self.device))
            self.logger.info(f"说话人分离模型已加载到 {self.device}",
                           extra_data={'device': self.device, 'model': self.model_name})
        except Exception as e:
            self.logger.error(f"模型加载失败: {str(e)}")
            raise

    def process(self, wav_file, num_speakers=None):
        """
        执行说话人分离

        Args:
            wav_file: 音频文件路径
            num_speakers: 说话人数量，如果为None则使用实例默认值或自动检测

        Returns:
            diarization: 分离结果对象
        """
        # 使用传入的参数或实例默认值
        effective_num_speakers = num_speakers or self.num_speakers

        try:
            # 加载音频文件
            waveform, sample_rate = torchaudio.load(wav_file)

            # 构建输入参数
            audio_input = {"waveform": waveform, "sample_rate": sample_rate}
            pipeline_kwargs = {}

            if effective_num_speakers is not None:
                pipeline_kwargs["num_speakers"] = effective_num_speakers

            # 执行说话人分离
            self.logger.debug(f"开始说话人分离: {wav_file}, 参数: {pipeline_kwargs}")
            diarization = self.pipeline(audio_input, **pipeline_kwargs)

            return diarization

        except Exception as e:
            self.logger.error(f"说话人分离失败: {wav_file}, 错误: {str(e)}")
            raise

    def save_rttm(self, diarization, rttm_file):
        """
        保存分离结果为RTTM文件

        Args:
            diarization: 分离结果对象
            rttm_file: RTTM文件路径
        """
        import os
        # 确保输出目录存在
        os.makedirs(os.path.dirname(rttm_file), exist_ok=True)

        with open(rttm_file, "w") as rttm:
            diarization.write_rttm(rttm)

    def check_rttm_exists(self, rttm_file):
        """
        检查RTTM文件是否已存在且有效

        Args:
            rttm_file: RTTM文件路径

        Returns:
            bool: 文件是否存在且有效
        """
        import os
        if not os.path.exists(rttm_file):
            return False

        # 检查文件是否为空
        if os.path.getsize(rttm_file) == 0:
            return False

        # 检查文件内容格式是否正确
        try:
            with open(rttm_file, 'r') as f:
                lines = f.readlines()
                if len(lines) == 0:
                    return False

                # 检查第一行格式是否正确
                first_line = lines[0].strip().split()
                if len(first_line) < 8 or first_line[0] != "SPEAKER":
                    return False

            return True
        except Exception:
            return False