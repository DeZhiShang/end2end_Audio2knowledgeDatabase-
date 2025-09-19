"""
音频切分模块
负责根据RTTM文件切分音频
"""

import warnings
# 过滤相关警告
warnings.filterwarnings("ignore", category=UserWarning)

import torchaudio
import os
from tqdm import tqdm


class AudioSegmentation:
    """音频切分处理器"""

    def __init__(self):
        """初始化音频切分器"""
        pass

    def parse_rttm_and_segment(self, rttm_file, wav_file, output_dir):
        """
        解析RTTM文件并根据时间戳切分音频

        Args:
            rttm_file: RTTM文件路径
            wav_file: 原始WAV音频文件路径
            output_dir: 输出目录路径
        """
        # 加载原始音频
        waveform, sample_rate = torchaudio.load(wav_file)

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 读取并解析RTTM文件内容
        segments = []
        with open(rttm_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 8:
                        start_time = float(parts[3])  # 第4列：起始时间
                        duration = float(parts[4])    # 第5列：持续时间
                        speaker_id = parts[7]         # 第8列：说话人ID

                        segments.append({
                            'start_time': start_time,
                            'duration': duration,
                            'speaker_id': speaker_id,
                            'end_time': start_time + duration
                        })

        # 按起始时间排序，确保处理顺序正确
        segments.sort(key=lambda x: x['start_time'])
        print(f"📊 读取到 {len(segments)} 个音频片段，按起始时间排序")

        # 使用tqdm显示音频切分进度
        segment_count = 0
        with tqdm(segments, desc="✂️ 切分音频片段", unit="片段") as pbar:
            for seg in pbar:
                start_time = seg['start_time']
                duration = seg['duration']
                speaker_id = seg['speaker_id']
                end_time = seg['end_time']

                # 验证时间逻辑
                if duration <= 0:
                    print(f"⚠️ 跳过无效时长片段: {start_time:.3f}s, 时长={duration:.3f}s")
                    continue

                # 计算样本索引
                start_sample = int(start_time * sample_rate)
                end_sample = int(end_time * sample_rate)

                # 确保不超出音频长度
                if start_sample >= waveform.shape[1]:
                    print(f"⚠️ 跳过超出音频长度的片段: {start_time:.3f}s")
                    continue

                if end_sample > waveform.shape[1]:
                    end_sample = waveform.shape[1]
                    end_time = end_sample / sample_rate

                # 确保起始样本小于结束样本
                if start_sample >= end_sample:
                    print(f"⚠️ 跳过无效样本范围: start={start_sample}, end={end_sample}")
                    continue

                # 切分音频片段
                segment = waveform[:, start_sample:end_sample]

                # 生成文件名：说话人-起始时间-结束时间.wav (使用序号确保有序)
                filename = f"{segment_count:03d}_{speaker_id}-{start_time:.3f}-{end_time:.3f}.wav"
                output_path = os.path.join(output_dir, filename)

                # 保存音频片段
                torchaudio.save(output_path, segment, sample_rate)
                segment_count += 1

                # 更新进度条信息
                pbar.set_postfix(
                    speaker=speaker_id,
                    start=f"{start_time:.2f}s",
                    duration=f"{duration:.2f}s",
                    refresh=True
                )

        print(f"成功切分 {segment_count} 个音频片段")