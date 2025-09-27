"""
ASR语音识别模块
负责使用SenseVoice-Small模型进行语音识别
"""

import warnings
# 过滤相关警告
warnings.filterwarnings("ignore", category=UserWarning)

import os
import glob
import re
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from src.utils.logger import get_logger


class ASRProcessor:
    """ASR语音识别处理器"""

    def __init__(self, model_path='/home/dzs-ai-4/dzs-dev/end2end_autio2kg/models/senseVoice-small'):
        """
        初始化ASR处理器

        Args:
            model_path: SenseVoice模型路径
        """
        self.logger = get_logger(__name__)
        self.model_path = model_path
        self.inference_pipeline = None
        self._initialize_model()

    def _initialize_model(self):
        """初始化SenseVoice推理管线"""
        try:
            self.inference_pipeline = pipeline(
                task=Tasks.auto_speech_recognition,
                model=self.model_path,
                model_revision="master",
                device="cuda:1",
                languege="zh"
            )
            self.logger.info("SenseVoice模型加载完成")
        except Exception as e:
            self.logger.error(f"SenseVoice模型初始化失败: {str(e)}")
            raise

    def extract_speaker_from_filename(self, filename):
        """
        从文件名中提取说话人信息

        Args:
            filename: 音频文件名，格式如 "000_SPEAKER_01-0.031-1.398.wav"

        Returns:
            str: 说话人ID，如 "SPEAKER_01"
        """
        # 使用正则表达式提取说话人ID
        match = re.search(r'SPEAKER_\d+', filename)
        if match:
            return match.group()
        return "UNKNOWN_SPEAKER"

    def clean_sensevoice_text(self, raw_text):
        """
        清理SenseVoice输出的文本，去除标记符号

        Args:
            raw_text: 原始文本，如 '<|zh|><|NEUTRAL|><|Speech|><|woitn|>喂你好'

        Returns:
            str: 清理后的纯文本，如 '喂你好'
        """
        if not raw_text:
            return ""

        # 使用正则表达式去除所有 <|...|> 格式的标记
        import re
        cleaned_text = re.sub(r'<\|[^|]*\|>', '', raw_text)
        return cleaned_text.strip()

    def process_single_audio(self, wav_file):
        """
        处理单个音频文件进行ASR识别

        Args:
            wav_file: 音频文件路径

        Returns:
            dict: 包含说话人信息和识别结果的字典
        """
        try:
            # 执行ASR识别
            rec_result = self.inference_pipeline(wav_file)

            # 提取说话人信息
            filename = os.path.basename(wav_file)
            speaker_id = self.extract_speaker_from_filename(filename)

            # 提取识别文本 - 处理SenseVoice的实际返回格式
            text = ""
            if isinstance(rec_result, list) and len(rec_result) > 0:
                # SenseVoice返回格式: [{'key': '...', 'text': '...'}]
                first_result = rec_result[0]
                if isinstance(first_result, dict) and 'text' in first_result:
                    raw_text = first_result['text']
                    text = self.clean_sensevoice_text(raw_text)
                else:
                    text = str(first_result)
            elif isinstance(rec_result, dict) and 'text' in rec_result:
                raw_text = rec_result['text']
                text = self.clean_sensevoice_text(raw_text)
            elif isinstance(rec_result, str):
                text = self.clean_sensevoice_text(rec_result)
            else:
                text = str(rec_result)

            return {
                'filename': filename,
                'speaker_id': speaker_id,
                'text': text.strip(),
                'success': True
            }

        except Exception as e:
            self.logger.error(f"ASR识别失败: {os.path.basename(wav_file)}")
            return {
                'filename': os.path.basename(wav_file),
                'speaker_id': self.extract_speaker_from_filename(os.path.basename(wav_file)),
                'text': f"[识别失败]",
                'success': False
            }

    def get_sorted_audio_files(self, audio_dir):
        """
        获取目录下所有音频文件并按序号排序

        Args:
            audio_dir: 音频文件目录

        Returns:
            list: 排序后的音频文件路径列表
        """
        # 获取所有wav文件
        audio_files = glob.glob(os.path.join(audio_dir, "*.wav"))

        # 按文件名中的序号排序（提取文件名开头的数字）
        def extract_sequence_number(filename):
            basename = os.path.basename(filename)
            match = re.match(r'(\d+)_', basename)
            return int(match.group(1)) if match else float('inf')

        audio_files.sort(key=extract_sequence_number)
        return audio_files

    def process_audio_directory(self, audio_dir, output_file, force_overwrite=False):
        """
        处理目录下的所有音频文件并生成markdown输出

        Args:
            audio_dir: 包含切分音频的目录
            output_file: 输出的markdown文件路径
            force_overwrite: 是否强制覆盖已存在的文件

        Returns:
            dict: 处理结果统计
        """
        # 检查是否已经处理过
        if not force_overwrite and os.path.exists(output_file):
            return {"success": 0, "error": 0, "skipped": 1, "total": 0}

        if not os.path.exists(audio_dir):
            self.logger.error(f"音频目录不存在: {audio_dir}")
            return {"success": 0, "error": 1, "skipped": 0, "total": 0}

        # 获取排序后的音频文件
        audio_files = self.get_sorted_audio_files(audio_dir)

        if not audio_files:
            self.logger.warning(f"{audio_dir} 目录下没有找到wav文件")
            return {"success": 0, "error": 0, "skipped": 0, "total": 0}

        # 创建输出目录
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        results = []
        success_count = 0
        error_count = 0

        # 处理音频文件
        for audio_file in audio_files:
            # 处理单个音频文件
            result = self.process_single_audio(audio_file)
            results.append(result)

            if result['success']:
                success_count += 1
            else:
                error_count += 1

        # 生成markdown内容
        self._generate_markdown_output(results, output_file, audio_dir)

        self.logger.info(f"ASR完成 - 成功: {success_count}, 失败: {error_count}")

        return {
            "success": success_count,
            "error": error_count,
            "skipped": 0,
            "total": len(audio_files)
        }

    def _generate_markdown_output(self, results, output_file, audio_dir):
        """
        生成markdown格式的输出文件

        Args:
            results: ASR识别结果列表
            output_file: 输出文件路径
            audio_dir: 音频目录路径
        """
        dir_name = os.path.basename(audio_dir)

        with open(output_file, 'w', encoding='utf-8') as f:
            # 写入文件头
            f.write(f"# {dir_name} - ASR识别结果\n\n")
            f.write(f"音频目录: `{audio_dir}`\n")
            f.write(f"处理时间: {self._get_current_time()}\n")
            f.write(f"总片段数: {len(results)}\n\n")
            f.write("---\n\n")

            # 写入每个识别结果
            for result in results:
                speaker_id = result['speaker_id']
                text = result['text']

                f.write(f"**{speaker_id}**: {text}\n\n")

    def _get_current_time(self):
        """获取当前时间字符串"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def check_asr_exists(self, output_file):
        """
        检查ASR结果是否已存在

        Args:
            output_file: 输出文件路径

        Returns:
            bool: ASR结果是否已存在
        """
        return os.path.exists(output_file)