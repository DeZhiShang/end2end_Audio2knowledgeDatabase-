"""
问答对抽取模块
从清洗后的客服对话中提取高质量问答对
"""

import os
import json
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime
import re
from src.core.prompt import get_qa_extraction_prompt

try:
    import openai
except ImportError:
    print("警告: openai包未安装，请运行 pip install openai")
    openai = None

try:
    from dotenv import load_dotenv
except ImportError:
    print("警告: python-dotenv包未安装，请运行 pip install python-dotenv")
    def load_dotenv():
        pass

# 加载环境变量
load_dotenv()

from src.utils.logger import get_logger
from src.core.knowledge_base import QAPair, get_knowledge_base
from src.utils.file_cleaner import get_file_cleaner


class QAExtractor:
    """问答对抽取器：从清洗后的对话中提取问答对"""

    def __init__(self, enable_auto_cleanup: bool = True, cleanup_dry_run: bool = False):
        """
        初始化问答对抽取器

        Args:
            enable_auto_cleanup: 是否启用自动清理中间文件
            cleanup_dry_run: 是否为清理干运行模式
        """
        self.logger = get_logger(__name__)

        if openai is None:
            raise ImportError("请先安装openai包: pip install openai")

        self.api_key = os.getenv('DASHSCOPE_API_KEY')
        self.base_url = os.getenv('DASHSCOPE_BASE_URL')

        if not self.api_key or not self.base_url:
            raise ValueError("请在.env文件中配置DASHSCOPE_API_KEY和DASHSCOPE_BASE_URL")

        # 配置OpenAI客户端（兼容阿里云DashScope API）
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

        self.model_name = "qwen-plus-latest"

        # 获取知识库实例
        self.knowledge_base = get_knowledge_base()

        # 文件清理器配置
        self.enable_auto_cleanup = enable_auto_cleanup
        self.cleanup_dry_run = cleanup_dry_run
        self.file_cleaner = None  # 延迟初始化

    def _initialize_file_cleaner(self):
        """延迟初始化文件清理器"""
        if self.file_cleaner is None and self.enable_auto_cleanup:
            try:
                self.file_cleaner = get_file_cleaner(
                    enable_cleanup=self.enable_auto_cleanup,
                    dry_run=self.cleanup_dry_run
                )
                mode_desc = "干运行模式" if self.cleanup_dry_run else "实际删除模式"
                self.logger.info(f"文件清理器初始化成功 ({mode_desc})")
            except Exception as e:
                self.logger.warning(f"文件清理器初始化失败: {str(e)}")
                self.file_cleaner = False  # 标记为失败

    def _trigger_file_cleanup(self, file_path: str) -> Dict[str, Any]:
        """
        触发文件清理

        Args:
            file_path: 触发清理的文件路径

        Returns:
            Dict[str, Any]: 清理结果
        """
        if not self.enable_auto_cleanup:
            return {
                "success": False,
                "message": "文件清理已禁用"
            }

        # 初始化清理器
        self._initialize_file_cleaner()
        if not self.file_cleaner or self.file_cleaner is False:
            return {
                "success": False,
                "error": "文件清理器不可用"
            }

        # 执行清理
        try:
            cleanup_result = self.file_cleaner.cleanup_intermediate_files(file_path)

            if cleanup_result["success"]:
                action_desc = "DRY-RUN清理" if self.cleanup_dry_run else "清理"
                self.logger.info(f"🧹 {action_desc}中间文件成功: {cleanup_result.get('file_number', 'unknown')}, "
                               f"释放空间: {cleanup_result.get('disk_space_freed', 0):.2f}MB")
            else:
                self.logger.warning(f"清理中间文件失败: {cleanup_result.get('error', '未知错误')}")

            return cleanup_result

        except Exception as e:
            self.logger.error(f"文件清理异常: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    def extract_qa_pairs_from_text(self, dialogue_content: str, source_file: str = "unknown") -> Dict[str, Any]:
        """
        从对话文本中抽取问答对

        Args:
            dialogue_content: 清洗后的对话内容
            source_file: 来源文件名

        Returns:
            Dict[str, Any]: 抽取结果
        """
        try:
            # 构建完整的prompt
            full_prompt = get_qa_extraction_prompt() + "\n" + dialogue_content

            # 调用LLM API
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": full_prompt}
                ],
                temperature=0.1,  # 较低的温度确保稳定输出
                max_tokens=32768,  # 足够的token数量
            )

            result_text = response.choices[0].message.content.strip()

            # 解析简化的QA对结果
            qa_pairs = self._parse_simple_qa_response(result_text, source_file)

            if qa_pairs:
                return {
                    "success": True,
                    "qa_pairs": qa_pairs,
                    "summary": {"total_pairs": len(qa_pairs)},
                    "original_content": dialogue_content,
                    "source_file": source_file,
                    "token_usage": {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens
                    },
                    "raw_response": result_text
                }
            else:
                return {
                    "success": False,
                    "error": "无法解析LLM响应",
                    "raw_response": result_text,
                    "source_file": source_file
                }

        except Exception as e:
            self.logger.error(f"问答对抽取失败: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "source_file": source_file
            }

    def _parse_simple_qa_response(self, response_text: str, source_file: str = "unknown") -> List[QAPair]:
        """
        解析简化的QA对响应

        Args:
            response_text: LLM响应文本

        Returns:
            List[QAPair]: 解析后的QA对列表
        """
        qa_pairs = []

        try:
            # 按Q:和A:模式解析
            lines = response_text.strip().split('\n')
            current_q = None
            current_a = None

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                if line.startswith('Q:'):
                    # 如果有未完成的QA对，先保存
                    if current_q and current_a:
                        qa_pair = QAPair(
                            id=str(uuid.uuid4()),
                            question=current_q.strip(),
                            answer=current_a.strip(),
                            source_file=source_file,
                            timestamp=datetime.now(),
                            metadata={
                                'category': 'unknown',
                                'keywords': [],
                                'confidence': 0.9,
                                'extraction_method': 'simple_llm'
                            }
                        )
                        qa_pairs.append(qa_pair)

                    # 开始新的问题
                    current_q = line[2:].strip()
                    current_a = None

                elif line.startswith('A:'):
                    # 记录答案
                    current_a = line[2:].strip()

                elif current_a is not None:
                    # 续接答案（多行答案情况）
                    current_a += " " + line
                elif current_q is not None:
                    # 续接问题（多行问题情况）
                    current_q += " " + line

            # 处理最后一个QA对
            if current_q and current_a:
                qa_pair = QAPair(
                    id=str(uuid.uuid4()),
                    question=current_q.strip(),
                    answer=current_a.strip(),
                    source_file=source_file,
                    timestamp=datetime.now(),
                    metadata={
                        'category': 'unknown',
                        'keywords': [],
                        'confidence': 0.9,
                        'extraction_method': 'simple_llm'
                    }
                )
                qa_pairs.append(qa_pair)

            pass  # 静默解析QA对
            return qa_pairs

        except Exception as e:
            self.logger.error(f"简化QA响应解析失败: {str(e)}")
            return []

    def extract_and_save_qa_pairs(self, input_file: str) -> Dict[str, Any]:
        """
        从文件中抽取问答对并保存到知识库

        Args:
            input_file: 输入的markdown文件路径

        Returns:
            Dict[str, Any]: 处理结果
        """
        if not os.path.exists(input_file):
            return {
                "success": False,
                "error": f"输入文件不存在: {input_file}"
            }

        try:
            # 读取对话内容
            with open(input_file, 'r', encoding='utf-8') as f:
                dialogue_content = f.read()

            filename = os.path.basename(input_file)
            self.logger.info(f"开始从 {filename} 抽取问答对...")

            # 抽取问答对
            extraction_result = self.extract_qa_pairs_from_text(dialogue_content, filename)

            if extraction_result["success"]:
                qa_pairs = extraction_result["qa_pairs"]

                if qa_pairs:
                    # 保存到知识库
                    success = self.knowledge_base.append_qa_pairs(qa_pairs)

                    if success:
                        # 更新文件状态
                        from src.core.knowledge_base import ProcessingStatus
                        self.knowledge_base.update_file_status(
                            input_file,
                            ProcessingStatus.QA_EXTRACTED,
                            {
                                'qa_count': len(qa_pairs),
                                'extraction_time': datetime.now().isoformat(),
                                'token_usage': extraction_result.get('token_usage', {})
                            }
                        )

                        self.logger.info(f"✅ 成功抽取并保存 {len(qa_pairs)} 个问答对")

                        # 触发中间文件清理
                        if self.enable_auto_cleanup:
                            try:
                                cleanup_result = self._trigger_file_cleanup(input_file)
                                # 清理结果记录在_trigger_file_cleanup中，这里不需要额外日志
                            except Exception as e:
                                self.logger.warning(f"文件清理触发失败: {str(e)}")

                        return {
                            "success": True,
                            "input_file": input_file,
                            "qa_count": len(qa_pairs),
                            "summary": extraction_result.get('summary', {}),
                            "token_usage": extraction_result.get('token_usage', {}),
                            "qa_pairs": qa_pairs
                        }
                    else:
                        return {
                            "success": False,
                            "error": "保存问答对到知识库失败",
                            "input_file": input_file
                        }
                else:
                    self.logger.warning(f"从 {filename} 中未抽取到任何问答对")
                    return {
                        "success": True,
                        "input_file": input_file,
                        "qa_count": 0,
                        "warning": "未抽取到问答对"
                    }
            else:
                return {
                    "success": False,
                    "error": extraction_result.get("error", "抽取失败"),
                    "input_file": input_file
                }

        except Exception as e:
            self.logger.error(f"处理文件 {input_file} 时出错: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "input_file": input_file
            }

    def batch_extract_qa_pairs(self, input_dir: str = "data/output/docs") -> Dict[str, Any]:
        """
        批量抽取目录下所有清洗完成文件的问答对

        Args:
            input_dir: 输入目录路径

        Returns:
            Dict[str, Any]: 批量处理结果统计
        """
        if not os.path.exists(input_dir):
            return {
                "success": False,
                "error": f"输入目录不存在: {input_dir}"
            }

        # 获取所有清洗完成的文件
        clean_finished_files = self.knowledge_base.get_clean_finished_files()

        if not clean_finished_files:
            self.logger.warning("没有找到状态为clean_finished的文件")
            return {
                "success": False,
                "error": "没有找到可处理的文件"
            }

        self.logger.info(f"🚀 开始批量问答对抽取，发现 {len(clean_finished_files)} 个已清洗文件")

        results = []
        success_count = 0
        error_count = 0
        total_qa_pairs = 0
        total_tokens = 0

        for file_path in clean_finished_files:
            self.logger.info(f"\n📄 处理文件: {os.path.basename(file_path)}")

            result = self.extract_and_save_qa_pairs(file_path)
            results.append(result)

            if result["success"]:
                success_count += 1
                qa_count = result.get("qa_count", 0)
                total_qa_pairs += qa_count

                token_usage = result.get("token_usage", {})
                total_tokens += token_usage.get("total_tokens", 0)

                self.logger.info(f"✅ 成功抽取 {qa_count} 个问答对")
            else:
                error_count += 1
                self.logger.error(f"❌ 处理失败: {result.get('error', '未知错误')}")

        # 统计总结
        self.logger.info(f"\n🎉 批量问答对抽取完成！")
        self.logger.info(f"✅ 成功: {success_count} 个文件")
        self.logger.info(f"❌ 失败: {error_count} 个文件")
        self.logger.info(f"📊 总计抽取: {total_qa_pairs} 个问答对")
        self.logger.info(f"🔢 总计消耗: {total_tokens} tokens")

        # 获取知识库统计
        kb_stats = self.knowledge_base.get_statistics()

        return {
            "success": True,
            "total_files": len(clean_finished_files),
            "success_count": success_count,
            "error_count": error_count,
            "total_qa_pairs": total_qa_pairs,
            "total_tokens": total_tokens,
            "results": results,
            "knowledge_base_stats": kb_stats
        }

    def get_extraction_statistics(self) -> Dict[str, Any]:
        """
        获取问答对抽取统计信息

        Returns:
            Dict[str, Any]: 统计信息
        """
        # 获取知识库统计
        kb_stats = self.knowledge_base.get_statistics()

        # 统计各种状态的文件数量
        status_counts = {}
        for file_path, file_status in self.knowledge_base.file_status_map.items():
            status = file_status.status.value
            status_counts[status] = status_counts.get(status, 0) + 1

        return {
            "knowledge_base_stats": kb_stats,
            "file_status_counts": status_counts,
            "ready_for_extraction": len(self.knowledge_base.get_clean_finished_files()),
            "total_tracked_files": len(self.knowledge_base.file_status_map)
        }