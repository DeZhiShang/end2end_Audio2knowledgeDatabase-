"""
LLM数据清洗模块
负责使用大语言模型清洗ASR识别结果，还原真实对话
"""

import os
from typing import Dict, Any
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


class LLMDataCleaner:
    """LLM数据清洗器：使用大语言模型清洗ASR识别结果"""

    def __init__(self):
        """初始化LLM清洗器"""
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

        # Gleaning机制配置
        self.max_gleaning_rounds = 3  # 最大清洗轮数
        self.quality_threshold = 0.90  # 质量阈值(0-1)
        # 移除最小改进阈值限制，以质量达标为准

    def get_cleaning_prompt(self) -> str:
        """
        获取数据清洗的prompt模板

        Returns:
            str: 完整的prompt模板
        """
        prompt = """你是一个专业的语音识别数据清洗专家，负责清洗博邦方舟无创血糖仪客服对话记录。

## 背景信息
- 对话场景：博邦方舟无创血糖仪客服与用户的电话咨询
- 语言：全部为中文对话
- 产品：博邦方舟无创血糖仪（非侵入式血糖检测设备）
- 数据来源：通过pyannote说话人分离 + SenseVoice ASR语音识别得到的文本

## 数据质量问题
由于技术限制，原始数据存在以下问题需要清洗：

1. **识别错误**：
   - "无川血糖仪"、"银行五创检查仪" → 应为"无创血糖仪"
   - "博邦方州"、"博帮方舟" → 应为"博邦方舟"
   - 日文误识别：语气词"额"、"嗯"被识别为日文"それ"等

2. **背景噪音干扰**：
   - 电视声音、短视频声音
   - 其他人说话声
   - 与血糖仪客服完全无关的内容

3. **语音质量问题**：
   - 模糊不清的语音
   - 重复、断续的表达
   - 口头禅和填充词过多

## 清洗要求

### 必须遵循的原则：
1. **真实性原则**：不得编造任何事实，宁缺毋滥
2. **专业性原则**：基于无创血糖仪客服场景进行判断
3. **保守性原则**：不确定的内容标注为"[不清楚]"，不要猜测

### 具体清洗标准：
1. **保留内容**：
   - 明确与血糖仪相关的对话
   - 客服专业回答
   - 用户的真实问题和反馈

2. **修正内容**：
   - 明显的产品名称识别错误
   - 常见的中文语音识别错误
   - 日文误识别的中文语气词

3. **删除内容**：
   - 明显的背景噪音（电视、音乐等）
   - 与血糖仪无关的对话内容
   - 重复的无意义语句

4. **标注内容**：
   - 不清楚的内容用"[不清楚]"标注
   - 推测性修正用"[疑似：原文]"标注

## 输出格式要求
请按照以下格式输出清洗后的对话：

```
# 清洗后的客服对话

**SPEAKER_00**: [清洗后的文本]

**SPEAKER_01**: [清洗后的文本]

**SPEAKER_00**: [清洗后的文本]

...

## 清洗说明
- 删除内容：[简要说明删除了什么内容]
- 修正内容：[说明主要修正了哪些识别错误]
- 不确定内容：[说明哪些内容标注为不清楚]
```

## 示例
原始ASR结果：
```
**SPEAKER_00**: 喂你好，这里是博帮方舟客服
**SPEAKER_01**: 你好我想咨询一下银行五创检查仪的使用方法
**SPEAKER_00**: それ好的，我来为您介绍无川血糖仪的使用
**SPEAKER_01**: 今天天气真好啊外面阳光明媚
```

清洗后结果：
```
# 清洗后的客服对话

**SPEAKER_00**: 喂你好，这里是博邦方舟客服

**SPEAKER_01**: 你好我想咨询一下无创血糖仪的使用方法

**SPEAKER_00**: 好的，我来为您介绍无创血糖仪的使用

## 清洗说明
- 删除内容：删除了与血糖仪无关的天气对话
- 修正内容：修正了"博帮方舟"→"博邦方舟"，"银行五创检查仪"→"无创血糖仪"，"无川血糖仪"→"无创血糖仪"，日文"それ"→删除
- 不确定内容：无
```

现在请清洗以下ASR识别结果：

"""
        return prompt

    def get_gleaning_prompt(self, round_number: int) -> str:
        """
        获取gleaning清洗的prompt模板（用于第二轮及后续清洗）

        Args:
            round_number: 当前清洗轮数

        Returns:
            str: gleaning prompt模板
        """
        prompt = f"""你是一个专业的语音识别数据清洗专家，正在进行第{round_number}轮精细化清洗。

## 任务背景
你正在对博邦方舟无创血糖仪客服对话进行多轮迭代清洗。前一轮清洗已经完成基础的错误修正和噪音过滤，现在需要进一步优化。

## 当前轮次目标
第{round_number}轮清洗重点关注：

### 1. 对话流畅性优化
- 检查对话是否自然流畅
- 修正语言表达的生硬之处
- 确保客服回答专业规范

### 2. 专业术语精准性
- 进一步检查血糖仪相关术语的准确性
- 统一产品名称的表述方式
- 确保技术描述的专业性

### 3. 逻辑一致性验证
- 检查对话逻辑是否合理
- 验证问答匹配度
- 确保时间顺序正确

### 4. 细节完善
- 补充可能遗漏的重要信息
- 删除仍然存在的冗余内容
- 优化语言表达的准确性

## 优化原则

### 核心要求：
1. **质量导向**：以对话质量达标为准，不设置最小改动限制
2. **保持真实**：绝不编造或添加不存在的内容
3. **专业标准**：确保客服对话达到专业服务标准
4. **渐进改进**：在前一轮基础上继续优化

### 具体标准：
- **语言流畅度**: 自然、专业、易懂
- **信息完整度**: 重要信息不缺失
- **逻辑清晰度**: 问答逻辑清楚
- **专业准确度**: 术语使用准确

## 输出要求

**重要**：无论内容质量如何，都必须输出完整的对话内容，按以下格式输出：

```
# 第{round_number}轮处理后的客服对话

**SPEAKER_00**: [处理后的对话内容，如无需修改则保持原内容]

**SPEAKER_01**: [处理后的对话内容，如无需修改则保持原内容]

## 第{round_number}轮处理说明
- 主要改进：[说明本轮的主要改进，如无改进则说明"内容质量已达标，保持原状"]
- 流畅性：[说明语言表达方面的处理]
- 专业性：[说明术语和表述方面的处理]
- 逻辑性：[说明逻辑结构方面的处理]

## 质量评估
- 流畅度：[1-10分评分]
- 专业度：[1-10分评分]
- 完整度：[1-10分评分]
- 准确度：[1-10分评分]
- 综合评分：[1-10分综合评分]
```

**注意**：即使内容已经很高质量，也必须完整输出对话内容，不能只输出评估信息。

现在请对以下对话进行第{round_number}轮精细化清洗：

"""
        return prompt

    def evaluate_content_quality(self, content: str) -> Dict[str, Any]:
        """
        评估内容质量，判断是否需要进一步清洗

        Args:
            content: 待评估的内容

        Returns:
            Dict[str, Any]: 质量评估结果
        """
        try:
            evaluation_prompt = """你是一个专业的对话质量评估专家。请对以下客服对话内容进行质量评估。

## 评估维度
1. **流畅度** (1-10分): 语言表达是否自然流畅
2. **专业度** (1-10分): 客服回答是否专业规范
3. **完整度** (1-10分): 重要信息是否完整
4. **准确度** (1-10分): 术语使用是否准确
5. **逻辑性** (1-10分): 对话逻辑是否清晰

## 输出格式
请严格按照以下JSON格式输出评估结果：
```json
{
    "fluency_score": 8,
    "professionalism_score": 7,
    "completeness_score": 9,
    "accuracy_score": 8,
    "logic_score": 8,
    "overall_score": 8.0,
    "needs_improvement": true,
    "improvement_suggestions": ["具体改进建议1", "具体改进建议2"]
}
```

待评估内容：
""" + content

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": evaluation_prompt}],
                temperature=0.1,
                max_tokens=1000,
            )

            result_text = response.choices[0].message.content.strip()

            # 尝试解析JSON结果
            import json
            import re

            # 提取JSON部分
            json_match = re.search(r'```json\n(.*?)\n```', result_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                evaluation_result = json.loads(json_str)
            else:
                # 如果没有找到JSON格式，尝试直接解析
                try:
                    evaluation_result = json.loads(result_text)
                except:
                    # 解析失败，返回默认评估
                    evaluation_result = {
                        "fluency_score": 7,
                        "professionalism_score": 7,
                        "completeness_score": 7,
                        "accuracy_score": 7,
                        "logic_score": 7,
                        "overall_score": 7.0,
                        "needs_improvement": True,
                        "improvement_suggestions": ["需要进一步优化"]
                    }

            return {
                "success": True,
                "evaluation": evaluation_result,
                "raw_response": result_text
            }

        except Exception as e:
            self.logger.warning(f"质量评估失败: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "evaluation": {
                    "overall_score": 5.0,
                    "needs_improvement": True,
                    "improvement_suggestions": ["评估失败，建议人工检查"]
                }
            }

    def clean_with_gleaning(self, asr_content: str, max_rounds: int = None, quality_threshold: float = None) -> Dict[str, Any]:
        """
        使用gleaning机制进行多轮迭代清洗

        Args:
            asr_content: ASR识别的原始内容
            max_rounds: 最大清洗轮数（None使用默认值）
            quality_threshold: 质量阈值（None使用默认值）

        Returns:
            Dict[str, Any]: 包含多轮清洗结果和统计信息的字典
        """
        # 使用默认配置
        max_rounds = max_rounds or self.max_gleaning_rounds
        quality_threshold = quality_threshold or self.quality_threshold

        pass  # 静默开始多轮清洗

        # 存储所有轮次的结果
        rounds_results = []
        current_content = asr_content
        total_tokens = 0

        try:
            # 第一轮：基础清洗
            first_round_result = self.clean_asr_result(current_content)

            if not first_round_result["success"]:
                return {
                    "success": False,
                    "error": f"第1轮清洗失败: {first_round_result.get('error', '未知错误')}",
                    "rounds": 1,
                    "total_tokens": 0
                }

            current_content = first_round_result["cleaned_content"]
            total_tokens += first_round_result["token_usage"]["total_tokens"]

            # 评估第一轮质量
            quality_eval = self.evaluate_content_quality(current_content)
            overall_score = quality_eval["evaluation"]["overall_score"] / 10.0 if quality_eval["success"] else 0.5

            rounds_results.append({
                "round": 1,
                "method": "basic_cleaning",
                "content": current_content,
                "quality_score": overall_score,
                "tokens": first_round_result["token_usage"]["total_tokens"],
                "evaluation": quality_eval
            })

            # 检查是否已达到质量要求
            if overall_score >= quality_threshold:
                return {
                    "success": True,
                    "rounds": 1,
                    "final_content": current_content,
                    "final_quality_score": overall_score,
                    "total_tokens": total_tokens,
                    "rounds_details": rounds_results,
                    "improvement_achieved": True,
                    "early_stop_reason": "quality_threshold_met"
                }

            # 后续轮次：gleaning优化
            previous_score = overall_score

            for round_num in range(2, max_rounds + 1):
                pass  # 静默Gleaning优化

                # 使用gleaning prompt
                gleaning_prompt = self.get_gleaning_prompt(round_num) + "\n" + current_content

                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": gleaning_prompt}],
                    temperature=0.1,
                    max_tokens=4000,
                )

                round_result = response.choices[0].message.content.strip()
                round_tokens = response.usage.total_tokens
                total_tokens += round_tokens

                # 评估本轮质量
                quality_eval = self.evaluate_content_quality(round_result)
                current_score = quality_eval["evaluation"]["overall_score"] / 10.0 if quality_eval["success"] else 0.5

                # 计算改进幅度
                improvement = current_score - previous_score

                rounds_results.append({
                    "round": round_num,
                    "method": "gleaning",
                    "content": round_result,
                    "quality_score": current_score,
                    "tokens": round_tokens,
                    "improvement": improvement,
                    "evaluation": quality_eval
                })

                pass  # 静默质量评分

                # 检查停止条件
                if current_score >= quality_threshold:
                    pass  # 静默达到质量阈值
                    return {
                        "success": True,
                        "rounds": round_num,
                        "final_content": round_result,
                        "final_quality_score": current_score,
                        "total_tokens": total_tokens,
                        "rounds_details": rounds_results,
                        "improvement_achieved": True,
                        "early_stop_reason": "quality_threshold_met"
                    }

                # 移除最小改进阈值检查，继续下一轮优化

                # 更新当前内容和分数
                current_content = round_result
                previous_score = current_score

            # 达到最大轮数，选择质量最高的轮次
            best_round = max(rounds_results, key=lambda r: r["quality_score"])
            self.logger.info(f"  🏁 达到最大轮数 ({max_rounds})，选择最佳结果 (第{best_round['round']}轮)")

            return {
                "success": True,
                "rounds": max_rounds,
                "final_content": best_round["content"],
                "final_quality_score": best_round["quality_score"],
                "total_tokens": total_tokens,
                "rounds_details": rounds_results,
                "improvement_achieved": best_round["quality_score"] > rounds_results[0]["quality_score"],
                "early_stop_reason": "max_rounds_reached"
            }

        except Exception as e:
            self.logger.error(f"Gleaning清洗失败: {str(e)}")
            # 如果有部分结果，返回最佳结果
            if rounds_results:
                best_round = max(rounds_results, key=lambda r: r["quality_score"])
                return {
                    "success": False,
                    "error": str(e),
                    "partial_success": True,
                    "rounds": len(rounds_results),
                    "final_content": best_round["content"],
                    "final_quality_score": best_round["quality_score"],
                    "total_tokens": total_tokens,
                    "rounds_details": rounds_results
                }
            else:
                return {
                    "success": False,
                    "error": str(e),
                    "rounds": 0,
                    "total_tokens": total_tokens
                }

    def clean_asr_result(self, asr_content: str) -> Dict[str, Any]:
        """
        使用LLM清洗ASR识别结果

        Args:
            asr_content: ASR识别的原始内容

        Returns:
            Dict[str, Any]: 包含清洗结果和元信息的字典
        """
        try:
            # 构建完整的prompt
            full_prompt = self.get_cleaning_prompt() + "\n" + asr_content

            # 调用LLM API
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": full_prompt}
                ],
                temperature=0.1,  # 较低的温度确保稳定输出
                max_tokens=4000,  # 足够的token数量
            )

            cleaned_content = response.choices[0].message.content.strip()

            return {
                "success": True,
                "original_content": asr_content,
                "cleaned_content": cleaned_content,
                "model": self.model_name,
                "token_usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }

        except Exception as e:
            self.logger.error(f"LLM清洗失败: {str(e)}")
            return {
                "success": False,
                "original_content": asr_content,
                "cleaned_content": asr_content,  # 失败时返回原内容
                "error": str(e),
                "model": self.model_name
            }

    def clean_markdown_file(self, input_file: str, output_file: str = None, enable_gleaning: bool = True, max_rounds: int = None, quality_threshold: float = None) -> Dict[str, Any]:
        """
        使用gleaning机制清洗markdown格式的ASR结果文件

        Args:
            input_file: 输入的markdown文件路径
            output_file: 输出的清洗后文件路径（默认为原文件名_gleaned.md）
            enable_gleaning: 是否启用gleaning多轮清洗
            max_rounds: 最大清洗轮数（None使用默认值）
            quality_threshold: 质量阈值（None使用默认值）

        Returns:
            Dict[str, Any]: 清洗结果统计
        """
        if not os.path.exists(input_file):
            return {
                "success": False,
                "error": f"输入文件不存在: {input_file}"
            }

        # 默认输出文件名
        if output_file is None:
            base_name = os.path.splitext(input_file)[0]
            suffix = "_gleaned" if enable_gleaning else "_cleaned"
            output_file = f"{base_name}{suffix}.md"

        try:
            # 读取原始ASR结果
            with open(input_file, 'r', encoding='utf-8') as f:
                original_content = f.read()

            self.logger.info(f"📖 读取ASR结果文件: {input_file}")
            self.logger.info(f"📄 原始内容长度: {len(original_content)} 字符")

            # 选择清洗方法
            if enable_gleaning:
                self.logger.info("🔄 使用Gleaning多轮清洗...")
                clean_result = self.clean_with_gleaning(original_content, max_rounds, quality_threshold)

                if clean_result["success"] or clean_result.get("partial_success"):
                    # 保存清洗后的结果
                    os.makedirs(os.path.dirname(output_file), exist_ok=True)
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(clean_result["final_content"])

                    self.logger.info(f"✅ Gleaning清洗完成，最终结果已保存至: {output_file}")
                    self.logger.info(f"📊 处理统计: {clean_result['rounds']}轮, {clean_result['total_tokens']} tokens")
                    self.logger.info(f"💯 最终质量评分: {clean_result['final_quality_score']:.2f}")

                    return {
                        "success": True,
                        "input_file": input_file,
                        "output_file": output_file,
                        "original_length": len(original_content),
                        "cleaned_length": len(clean_result["final_content"]),
                        "gleaning_enabled": True,
                        "rounds": clean_result["rounds"],
                        "final_quality_score": clean_result["final_quality_score"],
                        "total_tokens": clean_result["total_tokens"],
                        "improvement_achieved": clean_result.get("improvement_achieved", False),
                        "early_stop_reason": clean_result.get("early_stop_reason", "unknown"),
                        "rounds_details": clean_result.get("rounds_details", [])
                    }
                else:
                    self.logger.error(f"Gleaning清洗失败: {clean_result.get('error', '未知错误')}")
                    return {
                        "success": False,
                        "input_file": input_file,
                        "error": clean_result.get("error", "Gleaning清洗失败"),
                        "gleaning_enabled": True
                    }
            else:
                self.logger.info("🤖 使用标准单轮清洗...")
                clean_result = self.clean_asr_result(original_content)

                if clean_result["success"]:
                    # 保存清洗后的结果
                    os.makedirs(os.path.dirname(output_file), exist_ok=True)
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(clean_result["cleaned_content"])

                    self.logger.info(f"✅ 标准清洗完成，结果保存至: {output_file}")
                    self.logger.info(f"📊 Token使用情况: {clean_result['token_usage']['total_tokens']} tokens")

                    return {
                        "success": True,
                        "input_file": input_file,
                        "output_file": output_file,
                        "original_length": len(original_content),
                        "cleaned_length": len(clean_result["cleaned_content"]),
                        "gleaning_enabled": False,
                        "token_usage": clean_result["token_usage"]
                    }
                else:
                    self.logger.error(f"标准清洗失败: {clean_result.get('error', '未知错误')}")
                    return {
                        "success": False,
                        "input_file": input_file,
                        "error": clean_result.get("error", "标准清洗失败"),
                        "gleaning_enabled": False
                    }

        except Exception as e:
            self.logger.error(f"处理文件时出错: {str(e)}")
            return {
                "success": False,
                "input_file": input_file,
                "error": str(e)
            }


    def batch_clean_directory(self, input_dir: str = "docs", output_dir: str = "docs", enable_gleaning: bool = True, max_rounds: int = None, quality_threshold: float = None) -> Dict[str, Any]:
        """
        批量清洗目录下的所有ASR结果文件（支持gleaning）

        Args:
            input_dir: 输入目录路径
            output_dir: 输出目录路径
            enable_gleaning: 是否启用gleaning多轮清洗
            max_rounds: 最大清洗轮数（None使用默认值）
            quality_threshold: 质量阈值（None使用默认值）

        Returns:
            Dict[str, Any]: 批量处理结果统计
        """
        if not os.path.exists(input_dir):
            return {
                "success": False,
                "error": f"输入目录不存在: {input_dir}"
            }

        # 获取所有markdown文件
        md_files = [f for f in os.listdir(input_dir) if f.endswith('.md')]

        if not md_files:
            return {
                "success": False,
                "error": f"目录 {input_dir} 中没有找到markdown文件"
            }

        method_name = "Gleaning多轮清洗" if enable_gleaning else "标准清洗"
        pass  # 静默开始批量处理

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        results = []
        success_count = 0
        error_count = 0
        total_tokens = 0
        total_rounds = 0
        quality_scores = []

        for md_file in md_files:
            input_file = os.path.join(input_dir, md_file)
            output_file = os.path.join(output_dir, md_file)

            self.logger.info(f"\n📄 处理文件: {md_file}")
            result = self.clean_markdown_file(
                input_file, output_file, enable_gleaning, max_rounds, quality_threshold
            )
            results.append(result)

            if result["success"]:
                success_count += 1
                if enable_gleaning:
                    total_tokens += result["total_tokens"]
                    total_rounds += result["rounds"]
                    quality_scores.append(result["final_quality_score"])
                else:
                    total_tokens += result["token_usage"]["total_tokens"]
            else:
                error_count += 1

        # 统计总结
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        avg_rounds = total_rounds / success_count if success_count > 0 else 0

        self.logger.info(f"\n🎉 批量{method_name}完成！")
        self.logger.info(f"✅ 成功: {success_count} 个文件")
        self.logger.info(f"失败: {error_count} 个文件")
        self.logger.info(f"📊 总计使用: {total_tokens} tokens")

        if enable_gleaning and quality_scores:
            self.logger.info(f"💯 平均质量评分: {avg_quality:.2f}")
            self.logger.info(f"🔄 平均清洗轮数: {avg_rounds:.1f}")

        return {
            "success": True,
            "total_files": len(md_files),
            "success_count": success_count,
            "error_count": error_count,
            "total_tokens": total_tokens,
            "gleaning_enabled": enable_gleaning,
            "average_quality_score": avg_quality if enable_gleaning else None,
            "average_rounds": avg_rounds if enable_gleaning else None,
            "results": results,
            "output_directory": output_dir
        }



