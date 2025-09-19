"""
音频处理器
统一的音频处理流程管理
"""

import os
import glob
from tqdm import tqdm
from diarization import SpeakerDiarization
from audio_segmentation import AudioSegmentation


class AudioProcessor:
    """音频处理器：管理端到端的音频处理流程"""

    def __init__(self):
        """初始化处理器"""
        self.diarizer = SpeakerDiarization()
        self.segmenter = AudioSegmentation()

    def process_single_file(self, wav_file):
        """
        端到端处理单个音频文件：说话人分离 → 音频切分

        Args:
            wav_file: 音频文件路径

        Returns:
            bool: 处理是否成功
        """
        try:
            # 提取文件名（不含扩展名）
            filename = os.path.splitext(os.path.basename(wav_file))[0]

            print(f"\n🎵 开始处理: {wav_file}")

            # 1. 执行说话人分离
            print("  🔍 执行说话人分离...")
            diarization = self.diarizer.process(wav_file)

            # 2. 保存RTTM文件
            rttm_file = f"rttms/{filename}.rttm"
            self.diarizer.save_rttm(diarization, rttm_file)
            print(f"  💾 RTTM文件保存至: {rttm_file}")

            # 3. 执行音频切分
            print("  ✂️  开始音频切分...")
            output_directory = f"wavs/{filename}"
            self.segmenter.parse_rttm_and_segment(rttm_file, wav_file, output_directory)

            print(f"  ✅ {filename} 处理完成！")
            return True

        except Exception as e:
            print(f"  ❌ 处理 {wav_file} 时出错: {str(e)}")
            return False

    def process_batch(self, input_dir="wavs"):
        """
        批量处理指定目录下的所有音频文件

        Args:
            input_dir: 输入目录路径

        Returns:
            dict: 处理结果统计
        """
        # 获取目录下所有的wav文件
        wav_files = glob.glob(f"{input_dir}/*.wav")

        if not wav_files:
            print(f"⚠️  警告: {input_dir}目录下没有找到任何wav文件")
            return {"success": 0, "error": 0}

        print(f"🚀 发现 {len(wav_files)} 个音频文件，开始端到端批量处理...")
        print("流程: 音频加载 → 说话人分离 → RTTM保存 → 音频切分")

        success_count = 0
        error_count = 0

        # 使用tqdm显示批量处理进度
        with tqdm(wav_files, desc="🎵 处理音频文件", unit="文件") as pbar:
            for wav_file in pbar:
                # 检查音频文件是否存在
                if not os.path.exists(wav_file):
                    pbar.set_postfix(status="⚠️ 文件不存在", refresh=True)
                    error_count += 1
                    continue

                # 更新进度条显示当前文件
                filename = os.path.basename(wav_file)
                pbar.set_postfix(file=filename, refresh=True)

                # 执行端到端处理
                if self.process_single_file(wav_file):
                    success_count += 1
                    pbar.set_postfix(file=filename, status="✅ 完成", refresh=True)
                else:
                    error_count += 1
                    pbar.set_postfix(file=filename, status="❌ 失败", refresh=True)

        print(f"\n🎉 批量处理完成！")
        print(f"✅ 成功: {success_count} 个文件")
        print(f"❌ 失败: {error_count} 个文件")

        return {"success": success_count, "error": error_count}