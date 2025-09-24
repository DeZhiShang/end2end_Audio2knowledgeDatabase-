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


class QAExtractor:
    """问答对抽取器：从清洗后的对话中提取问答对"""

    def __init__(self):
        """初始化问答对抽取器"""
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

    def get_qa_extraction_prompt(self) -> str:
        """
        获取问答对抽取的prompt模板

        Returns:
            str: 完整的prompt模板
        """
        prompt = """你是一个专业的知识库构建专家，专门从博邦方舟无创血糖仪客服对话中抽取高质量问答对。

## 任务背景
- 目标：构建博邦方舟无创血糖仪的专业知识库
- 输入：已经LLM清洗过的高质量客服对话
- 输出：直接抽出QA对，只需要问题和答案

## 输出格式
直接输出问答对，每对之间用空行分隔，一定不能带有任何markdown字符，像"*,#这样的"，格式如下：

Q: 问题内容
A: 答案内容

Q: 问题内容
A: 答案内容

## 抽取原则

### 核心要求
1. **全面性**：确保不遗漏任何有价值的问答信息
2. **准确性**：基于对话事实，不得编造或添加不存在的内容
3. **专业性**：问答对应体现专业客服水准
4. **实用性**：问答对应对实际用户有帮助

### 问答对质量标准
1. **问题明确**：问题表述清晰、具体、易理解
2. **答案完整**：答案准确、详细、有帮助
3. **逻辑清晰**：问答逻辑对应，避免答非所问
4. **语言规范**：使用标准、专业的表达方式

## 抽取策略

### 1. 直接问答对
- 用户明确提问，客服直接回答的内容
- 保持问题的原始意图，优化语言表达

### 2. 隐含问答对
- 从对话中推导出的常见问题和答案
- 基于用户需求和客服解释生成问答对

### 3. 知识点问答对
- 客服主动介绍的产品知识点
- 转换为问答形式，便于知识库查询

### 4. 问题细分
- 复杂问题拆分为多个简单问答对
- 每个问答对聚焦一个具体知识点

## 领域范围
需要涵盖但不限于以下领域：

### 产品基础信息
- 产品介绍、功能特点、技术原理
- 产品规格、型号、配置信息
- 适用人群、使用场景

### 使用操作
- 开机设置、基本操作流程
- 测量方法、注意事项
- 数据查看、记录管理

### 故障解决
- 常见问题、故障现象
- 解决方法、操作步骤
- 预防措施、维护建议

### 购买咨询
- 价格信息、购买渠道
- 优惠活动、促销政策
- 售后服务、质保信息

### 技术支持
- 产品原理、技术细节
- 与其他设备对比
- 专业术语解释


## 注意事项

### 质量控制
1. **置信度要求**：只输出置信度≥0.8的问答对，确保问答对能作为知识库的高质量语料
2. **去重处理**：避免重复或高度相似的问答对
3. **完整性检查**：确保答案完整，避免截断
4. **专业性验证**：答案应体现专业客服水准

### 语言规范
1. **标准表达**：使用规范的产品名称和术语
2. **用户视角**：问题从用户角度表述
3. **服务语调**：答案保持专业友好的服务语调
4. **避免口语化**：减少"嗯"、"哦"等口语化表达

## 示例

输入对话：
```
**SPEAKER_00**: 你好，我想了解一下博邦方舟无创血糖仪的工作原理  
**SPEAKER_01**: 您好！博邦方舟无创血糖仪采用代谢热整合法进行检测。人体代谢产生的热量与血糖浓度、耗氧量密切相关。当人体和环境处于热平衡状态时，代谢产生的热量与散发到环境的热量基本相等。设备通过多传感器夹子式探头，监测环境温度、湿度、人体温度、血流速、辐射散热量、蒸发散热量、血氧饱和度等参数，据此计算出血糖水平。  
**SPEAKER_00**: 那准确度怎么样？  
**SPEAKER_01**: 产品在2018年完成了注册临床试验（北京大学第一医院、北京协和医院、应急总医院），检测结果与静脉血检测对比一致性达93.9%，与指尖血检测对比一致性为94.4%，符合临床应用标准。 
```

输出示例：
Q: 博邦方舟无创血糖仪的工作原理是什么？
A: 博邦方舟无创血糖仪基于代谢热整合法原理进行血糖检测。研究表明，人体代谢产生的热量与血糖浓度、耗氧量正相关。当人体与环境处于热平衡状态时，代谢热量与散发的热量基本相等。设备通过夹子式探头集成多种传感器，监测环境温度、湿度、人体温度、血流速、辐射散热量、蒸发散热量和血氧饱和度等指标，并结合代谢热量和耗氧量计算出血糖浓度。夹子式结构设计能保证检测过程的稳定性和准确性。

Q: 博邦方舟无创血糖仪的测量准确性依据是什么？
A: 本产品于2018年在北京大学第一医院、北京协和医院和应急总医院完成注册临床试验。检测结果显示，与静脉血检测相比一致性为93.9%；与指尖血检测相比一致性为94.4%。这些结果表明设备的检测准确度符合临床应用标准，能够满足日常血糖监测需求。


现在请对以下清洗后的客服对话进行问答对抽取：

"""
        return prompt

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
            full_prompt = self.get_qa_extraction_prompt() + "\n" + dialogue_content

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