"""
音频格式转换器
支持MP3转WAV格式转换，使用GPU加速
"""

import os
import glob
import torch
import torchaudio
from pathlib import Path
from src.utils.logger import get_logger


class AudioConverter:
    """音频格式转换器：支持MP3转WAV"""

    def __init__(self, device="cuda:0"):
        """
        初始化音频转换器

        Args:
            device: 设备类型，默认使用cuda:0
        """
        self.logger = get_logger(__name__)
        self.device = device if torch.cuda.is_available() else "cpu"
        self.logger.info(f"音频转换器初始化完成，使用设备: {self.device}", extra_data={'device': self.device})

    def convert_single_file(self, input_file, output_file, target_sample_rate=16000):
        """
        转换单个音频文件

        Args:
            input_file: 输入文件路径（MP3）
            output_file: 输出文件路径（WAV）
            target_sample_rate: 目标采样率，默认16kHz

        Returns:
            bool: 转换是否成功
        """
        try:
            # 加载音频文件，自动归一化
            waveform, sample_rate = torchaudio.load(input_file, normalize=True)

            # 移动到GPU（如果可用）
            if self.device.startswith("cuda"):
                waveform = waveform.to(self.device)

            # 重采样到目标采样率
            if sample_rate != target_sample_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate,
                    new_freq=target_sample_rate
                ).to(self.device)
                waveform = resampler(waveform)

            # 移回CPU进行保存
            waveform = waveform.cpu()

            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_file), exist_ok=True)

            # 保存为WAV格式
            torchaudio.save(output_file, waveform, target_sample_rate)

            return True

        except Exception as e:
            self.logger.error(f"转换文件 {input_file} 时出错: {str(e)}", extra_data={'file': input_file, 'error': str(e)})
            return False

    def convert_mp3_to_wav(self, input_dir="mp3s", output_dir="wavs", target_sample_rate=16000):
        """
        批量转换MP3文件到WAV格式

        Args:
            input_dir: MP3文件所在目录
            output_dir: WAV文件输出目录
            target_sample_rate: 目标采样率，默认16kHz

        Returns:
            dict: 转换结果统计
        """
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

        # 获取所有MP3文件
        mp3_files = glob.glob(f"{input_dir}/*.mp3")

        if not mp3_files:
            self.logger.warning(f"警告: {input_dir}目录下没有找到任何MP3文件")
            return {"success": 0, "error": 0, "skipped": 0}

        self.logger.info(f"发现 {len(mp3_files)} 个MP3文件，开始批量转换...", extra_data={'file_count': len(mp3_files)})
        self.logger.info(f"转换流程: MP3 → 加载到{self.device} → 重采样到{target_sample_rate}Hz → 保存为WAV")

        success_count = 0
        error_count = 0
        skipped_count = 0

        # 批量转换音频文件
        for mp3_file in mp3_files:
            # 生成对应的WAV文件路径
            filename = os.path.splitext(os.path.basename(mp3_file))[0]
            wav_file = os.path.join(output_dir, f"{filename}.wav")

            # 检查WAV文件是否已存在
            if os.path.exists(wav_file):
                self.logger.info(f"跳过已存在的文件: {wav_file}")
                skipped_count += 1
                continue

            # 执行转换
            if self.convert_single_file(mp3_file, wav_file, target_sample_rate):
                success_count += 1
                self.logger.info(f"转换完成: {mp3_file} → {wav_file}")
            else:
                error_count += 1

        self.logger.info("批量转换完成！")
        self.logger.info(f"转换结果统计 - 成功: {success_count}个, 失败: {error_count}个, 跳过: {skipped_count}个",
                        extra_data={'success_count': success_count, 'error_count': error_count, 'skipped_count': skipped_count})

        return {
            "success": success_count,
            "error": error_count,
            "skipped": skipped_count
        }

    def get_audio_info(self, audio_file):
        """
        获取音频文件信息

        Args:
            audio_file: 音频文件路径

        Returns:
            dict: 音频信息
        """
        try:
            metadata = torchaudio.info(audio_file)
            return {
                "sample_rate": metadata.sample_rate,
                "num_frames": metadata.num_frames,
                "num_channels": metadata.num_channels,
                "duration": metadata.num_frames / metadata.sample_rate,
                "encoding": metadata.encoding
            }
        except Exception as e:
            self.logger.error(f"获取音频信息失败: {str(e)}", extra_data={'file': audio_file, 'error': str(e)})
            return None