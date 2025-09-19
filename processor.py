"""
音频处理器
统一的音频处理流程管理
"""

import os
import glob
from tqdm import tqdm
from diarization import SpeakerDiarization
from audio_segmentation import AudioSegmentation
from audio_converter import AudioConverter
from asr import ASRProcessor


class AudioProcessor:
    """音频处理器：管理端到端的音频处理流程"""

    def __init__(self):
        """初始化处理器"""
        self.converter = AudioConverter()
        self.diarizer = SpeakerDiarization()
        self.segmenter = AudioSegmentation()
        self.asr_processor = ASRProcessor()

    def process_single_file(self, wav_file, force_overwrite=False):
        """
        端到端处理单个音频文件：说话人分离 → 音频切分 → ASR识别

        Args:
            wav_file: 音频文件路径
            force_overwrite: 是否强制覆盖已存在的结果

        Returns:
            str: 处理结果状态 ("success", "error", "skipped")
        """
        try:
            # 提取文件名（不含扩展名）
            filename = os.path.splitext(os.path.basename(wav_file))[0]
            print(f"\n🎵 开始处理: {wav_file}")

            # 检查是否完全跳过（三个步骤都已完成）
            rttm_file = f"rttms/{filename}.rttm"
            output_directory = f"wavs/{filename}"
            asr_output_file = f"docs/{filename}.md"

            rttm_exists = self.diarizer.check_rttm_exists(rttm_file)
            segmentation_exists = self.segmenter.check_segmentation_exists(output_directory)
            asr_exists = self.asr_processor.check_asr_exists(asr_output_file)

            if not force_overwrite and rttm_exists and segmentation_exists and asr_exists:
                wav_files = [f for f in os.listdir(output_directory) if f.endswith('.wav')] if os.path.exists(output_directory) else []
                print(f"  ⏭️  完全跳过：所有步骤均已完成，发现{len(wav_files)}个片段，ASR结果: {asr_output_file}")
                return "skipped"

            # 1. 检查并执行说话人分离
            if not force_overwrite and rttm_exists:
                print(f"  ⏭️  跳过已存在的说话人分离结果: {rttm_file}")
            else:
                print("  🔍 执行说话人分离...")
                diarization = self.diarizer.process(wav_file)
                self.diarizer.save_rttm(diarization, rttm_file)
                print(f"  💾 RTTM文件保存至: {rttm_file}")

            # 2. 检查并执行音频切分
            if not force_overwrite and segmentation_exists:
                wav_files = [f for f in os.listdir(output_directory) if f.endswith('.wav')]
                print(f"  ⏭️  跳过已存在的音频切分结果，发现{len(wav_files)}个片段")
            else:
                print("  ✂️  开始音频切分...")
                self.segmenter.parse_rttm_and_segment(rttm_file, wav_file, output_directory, force_overwrite)

            # 3. 检查并执行ASR识别
            if not force_overwrite and asr_exists:
                print(f"  ⏭️  跳过已存在的ASR识别结果: {asr_output_file}")
            else:
                print("  🎙️  开始ASR语音识别...")
                # 确保docs目录存在
                os.makedirs("docs", exist_ok=True)
                asr_result = self.asr_processor.process_audio_directory(output_directory, asr_output_file, force_overwrite)
                print(f"  📝 ASR识别完成: 成功{asr_result['success']}个, 失败{asr_result['error']}个")

            print(f"  ✅ {filename} 完整处理完成！")
            return "success"

        except Exception as e:
            print(f"  ❌ 处理 {wav_file} 时出错: {str(e)}")
            return "error"

    def convert_mp3_to_wav(self, input_dir="mp3s", output_dir="wavs"):
        """
        批量转换MP3文件为WAV格式

        Args:
            input_dir: MP3文件所在目录
            output_dir: WAV文件输出目录

        Returns:
            dict: 转换结果统计
        """
        print("🔄 开始MP3转WAV预处理...")
        return self.converter.convert_mp3_to_wav(input_dir, output_dir)

    def process_batch(self, input_dir="wavs", enable_mp3_conversion=True, force_overwrite=False):
        """
        批量处理指定目录下的所有音频文件
        支持自动MP3转WAV预处理和智能跳过

        Args:
            input_dir: 输入目录路径
            enable_mp3_conversion: 是否启用MP3转WAV预处理
            force_overwrite: 是否强制覆盖已存在的结果

        Returns:
            dict: 处理结果统计
        """
        # 步骤1: 如果启用了MP3转换，先执行转换
        if enable_mp3_conversion:
            conversion_results = self.convert_mp3_to_wav()
            print(f"📋 MP3转换结果: 成功{conversion_results['success']}个, "
                  f"失败{conversion_results['error']}个, "
                  f"跳过{conversion_results['skipped']}个")

        # 步骤2: 获取目录下所有的wav文件
        wav_files = glob.glob(f"{input_dir}/*.wav")

        if not wav_files:
            print(f"⚠️  警告: {input_dir}目录下没有找到任何wav文件")
            return {"success": 0, "error": 0}

        print(f"🚀 发现 {len(wav_files)} 个音频文件，开始端到端批量处理...")
        if force_overwrite:
            print("⚠️  强制覆盖模式：重新处理所有文件")
        else:
            print("🧠 智能跳过模式：跳过已处理的文件")
        print("流程: 音频加载 → 说话人分离 → RTTM保存 → 音频切分")

        success_count = 0
        error_count = 0
        skipped_count = 0

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
                result = self.process_single_file(wav_file, force_overwrite)
                if result == "success":
                    success_count += 1
                    pbar.set_postfix(file=filename, status="✅ 完成", refresh=True)
                elif result == "skipped":
                    skipped_count += 1
                    pbar.set_postfix(file=filename, status="⏭️ 跳过", refresh=True)
                else:  # error
                    error_count += 1
                    pbar.set_postfix(file=filename, status="❌ 失败", refresh=True)

        print(f"\n🎉 批量处理完成！")
        print(f"✅ 成功: {success_count} 个文件")
        print(f"⏭️  跳过: {skipped_count} 个文件")
        print(f"❌ 失败: {error_count} 个文件")

        return {"success": success_count, "error": error_count, "skipped": skipped_count}