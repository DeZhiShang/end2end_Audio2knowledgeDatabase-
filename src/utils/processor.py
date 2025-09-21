"""
音频处理器
统一的音频处理流程管理
"""

import os
import glob
from tqdm import tqdm
from src.core.diarization import SpeakerDiarization
from src.core.audio_segmentation import AudioSegmentation
from src.utils.audio_converter import AudioConverter
from src.core.asr import ASRProcessor
from src.core.llm_cleaner import LLMDataCleaner
from src.utils.logger import get_logger


class AudioProcessor:
    """音频处理器：管理端到端的音频处理流程"""

    def __init__(self):
        """初始化处理器"""
        self.logger = get_logger(__name__)
        self.converter = AudioConverter()
        self.diarizer = SpeakerDiarization()
        self.segmenter = AudioSegmentation()
        self.asr_processor = ASRProcessor()
        self.llm_cleaner = None  # 延迟初始化LLM清洗器

        # Gleaning机制配置
        self.enable_gleaning = True  # 默认启用gleaning多轮清洗
        self.max_gleaning_rounds = 3  # 最大清洗轮数
        # 注意：质量阈值由LLMDataCleaner管理，避免重复配置

    def _initialize_llm_cleaner(self):
        """延迟初始化LLM清洗器"""
        if self.llm_cleaner is None:
            try:
                self.llm_cleaner = LLMDataCleaner()
                self.logger.info("LLM清洗器初始化成功")
            except Exception as e:
                self.logger.warning(f"LLM清洗器初始化失败: {str(e)}")
                self.llm_cleaner = False  # 标记为失败，避免重复尝试

    def process_single_file(self, wav_file, force_overwrite=False, enable_llm_cleaning=True, enable_gleaning=None):
        """
        端到端处理单个音频文件：说话人分离 → 音频切分 → ASR识别 → LLM清洗

        Args:
            wav_file: 音频文件路径
            force_overwrite: 是否强制覆盖已存在的结果
            enable_llm_cleaning: 是否启用LLM数据清洗
            enable_gleaning: 是否启用gleaning多轮清洗（None使用默认配置）

        Returns:
            str: 处理结果状态 ("success", "error", "skipped")
        """
        # 使用默认gleaning配置
        if enable_gleaning is None:
            enable_gleaning = self.enable_gleaning
        try:
            # 提取文件名（不含扩展名）
            filename = os.path.splitext(os.path.basename(wav_file))[0]
            self.logger.info(f"开始处理音频文件: {wav_file}")

            # 检查是否完全跳过（四个步骤都已完成）
            rttm_file = f"data/processed/rttms/{filename}.rttm"
            output_directory = f"data/processed/wavs/{filename}"
            asr_output_file = f"data/output/docs/{filename}.md"

            # 统一输出到docs目录（ASR和LLM清洗都用同一文件）
            final_output_file = asr_output_file  # 无论是否清洗，都输出到同一文件

            rttm_exists = self.diarizer.check_rttm_exists(rttm_file)
            segmentation_exists = self.segmenter.check_segmentation_exists(output_directory)
            asr_exists = self.asr_processor.check_asr_exists(asr_output_file)
            # 检查最终文件是否存在（不区分ASR或清洗后）
            final_exists = os.path.exists(final_output_file)

            # 完全跳过条件：所有必要步骤都已完成，最终文件存在
            skip_condition = (rttm_exists and segmentation_exists and final_exists)

            if not force_overwrite and skip_condition:
                # 更严谨的检查：确保音频切分目录存在且包含文件
                if os.path.exists(output_directory):
                    wav_files = [f for f in os.listdir(output_directory) if f.endswith('.wav')]
                    file_count_msg = f"发现{len(wav_files)}个片段"
                else:
                    file_count_msg = "音频目录不存在但ASR结果存在"

                method_desc = "Gleaning清洗" if enable_llm_cleaning and enable_gleaning else ("标准清洗" if enable_llm_cleaning else "ASR识别")
                self.logger.info(f"完全跳过：所有步骤均已完成，{file_count_msg}，最终结果: {final_output_file} ({method_desc})")
                return "skipped"

            # 1. 检查并执行说话人分离
            if not force_overwrite and rttm_exists:
                self.logger.info(f"跳过已存在的说话人分离结果: {rttm_file}")
            else:
                self.logger.info("执行说话人分离...")
                diarization = self.diarizer.process(wav_file)
                self.diarizer.save_rttm(diarization, rttm_file)
                self.logger.info(f"RTTM文件保存至: {rttm_file}")

            # 2. 检查并执行音频切分
            if not force_overwrite and segmentation_exists:
                wav_files = [f for f in os.listdir(output_directory) if f.endswith('.wav')]
                self.logger.info(f"跳过已存在的音频切分结果，发现{len(wav_files)}个片段")
            else:
                self.logger.info("开始音频切分...")
                self.segmenter.parse_rttm_and_segment(rttm_file, wav_file, output_directory, force_overwrite)

            # 3. 检查并执行ASR识别
            if not force_overwrite and asr_exists:
                self.logger.info(f"跳过已存在的ASR识别结果: {asr_output_file}")
            else:
                self.logger.info("开始ASR语音识别...")
                # 确保docs目录存在
                os.makedirs("data/output/docs", exist_ok=True)
                asr_result = self.asr_processor.process_audio_directory(output_directory, asr_output_file, force_overwrite)
                self.logger.info(f"ASR识别完成: 成功{asr_result['success']}个, 失败{asr_result['error']}个",
                               extra_data={'success_count': asr_result['success'], 'error_count': asr_result['error']})

            # 4. 检查并执行LLM数据清洗（直接覆盖ASR文件）
            if enable_llm_cleaning:
                self.logger.info("开始数据清洗并覆盖ASR结果...")
                self._initialize_llm_cleaner()
                if self.llm_cleaner and self.llm_cleaner is not False:
                    if enable_gleaning:
                        self.logger.info("使用Gleaning多轮清洗...")
                        clean_result = self.llm_cleaner.clean_markdown_file(
                            asr_output_file,
                            asr_output_file,  # 直接覆盖原文件
                            enable_gleaning=True,
                            max_rounds=self.max_gleaning_rounds
                            # 使用LLMDataCleaner的默认质量阈值
                        )
                        if clean_result["success"]:
                            self.logger.info(f"Gleaning清洗完成: {clean_result['rounds']}轮, {clean_result['total_tokens']} tokens, 质量评分: {clean_result['final_quality_score']:.2f}",
                                           extra_data={'rounds': clean_result['rounds'], 'tokens': clean_result['total_tokens'], 'quality_score': clean_result['final_quality_score']})
                        else:
                            self.logger.error(f"Gleaning清洗失败: {clean_result.get('error', '未知错误')}")
                    else:
                        self.logger.info("使用标准清洗...")
                        clean_result = self.llm_cleaner.clean_markdown_file(
                            asr_output_file,
                            asr_output_file,  # 直接覆盖原文件
                            enable_gleaning=False
                        )
                        if clean_result["success"]:
                            if 'total_tokens' in clean_result:
                                self.logger.info(f"标准清洗完成: {clean_result['total_tokens']} tokens",
                                               extra_data={'tokens': clean_result['total_tokens']})
                            else:
                                self.logger.info("标准清洗完成")
                        else:
                            self.logger.error(f"标准清洗失败: {clean_result.get('error', '未知错误')}")
                else:
                    self.logger.warning("LLM清洗器不可用，跳过数据清洗")

            self.logger.info(f"{filename} 完整处理完成！")
            return "success"

        except Exception as e:
            self.logger.error(f"处理 {wav_file} 时出错: {str(e)}", extra_data={'file': wav_file, 'error': str(e)})
            return "error"

    def convert_mp3_to_wav(self, input_dir="data/input/mp3s", output_dir="data/processed/wavs"):
        """
        批量转换MP3文件为WAV格式

        Args:
            input_dir: MP3文件所在目录
            output_dir: WAV文件输出目录

        Returns:
            dict: 转换结果统计
        """
        self.logger.info("开始MP3转WAV预处理...")
        return self.converter.convert_mp3_to_wav(input_dir, output_dir)

    def process_batch(self, input_dir="data/processed/wavs", enable_mp3_conversion=True, force_overwrite=False, enable_llm_cleaning=True, enable_gleaning=None):
        """
        批量处理指定目录下的所有音频文件
        支持自动MP3转WAV预处理、智能跳过和LLM数据清洗（含Gleaning）

        Args:
            input_dir: 输入目录路径
            enable_mp3_conversion: 是否启用MP3转WAV预处理
            force_overwrite: 是否强制覆盖已存在的结果
            enable_llm_cleaning: 是否启用LLM数据清洗
            enable_gleaning: 是否启用gleaning多轮清洗（None使用默认配置）

        Returns:
            dict: 处理结果统计
        """
        # 使用默认gleaning配置
        if enable_gleaning is None:
            enable_gleaning = self.enable_gleaning
        # 步骤1: 如果启用了MP3转换，先执行转换
        if enable_mp3_conversion:
            conversion_results = self.convert_mp3_to_wav()
            self.logger.info(f"MP3转换结果: 成功{conversion_results['success']}个, "
                           f"失败{conversion_results['error']}个, "
                           f"跳过{conversion_results['skipped']}个",
                           extra_data=conversion_results)

        # 步骤2: 获取目录下所有的wav文件
        wav_files = glob.glob(f"{input_dir}/*.wav")

        if not wav_files:
            self.logger.warning(f"警告: {input_dir}目录下没有找到任何wav文件")
            return {"success": 0, "error": 0}

        self.logger.info(f"发现 {len(wav_files)} 个音频文件，开始端到端批量处理...", extra_data={'file_count': len(wav_files)})
        if force_overwrite:
            self.logger.warning("强制覆盖模式：重新处理所有文件")
        else:
            self.logger.info("智能跳过模式：跳过已处理的文件")

        if enable_llm_cleaning:
            if enable_gleaning:
                llm_status = "→ Gleaning多轮清洗"
            else:
                llm_status = "→ 标准清洗"
        else:
            llm_status = ""

        self.logger.info(f"处理流程: MP3转WAV → 说话人分离 → 音频切分 → ASR识别{llm_status} → 高质量语料")

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
                result = self.process_single_file(wav_file, force_overwrite, enable_llm_cleaning, enable_gleaning)
                if result == "success":
                    success_count += 1
                    pbar.set_postfix(file=filename, status="✅ 完成", refresh=True)
                elif result == "skipped":
                    skipped_count += 1
                    pbar.set_postfix(file=filename, status="⏭️ 跳过", refresh=True)
                else:  # error
                    error_count += 1
                    pbar.set_postfix(file=filename, status="❌ 失败", refresh=True)

        self.logger.info("批量处理完成！")
        self.logger.info(f"处理结果统计 - 成功: {success_count}个, 跳过: {skipped_count}个, 失败: {error_count}个",
                        extra_data={'success': success_count, 'skipped': skipped_count, 'error': error_count})

        return {"success": success_count, "error": error_count, "skipped": skipped_count}