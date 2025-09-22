"""
智能Compact压缩系统
实现问答对的去重、合并和质量优化
"""

import os
import json
import uuid
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime
from collections import defaultdict
import re
from difflib import SequenceMatcher
import threading

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
from src.core.knowledge_base import QAPair


class QASimilarityAnalyzer:
    """问答对相似性分析器"""

    def __init__(self):
        self.logger = get_logger(__name__)

    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        计算两个文本的相似度

        Args:
            text1: 文本1
            text2: 文本2

        Returns:
            float: 相似度分数 (0.0-1.0)
        """
        # 基于序列匹配的相似度
        similarity = SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
        return similarity

    def calculate_semantic_similarity(self, qa1: QAPair, qa2: QAPair) -> float:
        """
        计算两个问答对的语义相似度

        Args:
            qa1: 问答对1
            qa2: 问答对2

        Returns:
            float: 语义相似度分数 (0.0-1.0)
        """
        # 问题相似度 (权重: 0.6)
        question_sim = self.calculate_text_similarity(qa1.question, qa2.question)

        # 答案相似度 (权重: 0.4)
        answer_sim = self.calculate_text_similarity(qa1.answer, qa2.answer)

        # 关键词重叠度 (权重: 0.3)
        keywords1 = set(qa1.metadata.get('keywords', []))
        keywords2 = set(qa2.metadata.get('keywords', []))
        keyword_overlap = 0.0
        if keywords1 and keywords2:
            intersection = len(keywords1.intersection(keywords2))
            union = len(keywords1.union(keywords2))
            keyword_overlap = intersection / union if union > 0 else 0.0

        # 分类匹配度 (权重: 0.2)
        category_match = 1.0 if qa1.metadata.get('category') == qa2.metadata.get('category') else 0.0

        # 综合相似度计算
        semantic_similarity = (
            question_sim * 0.6 +
            answer_sim * 0.4 +
            keyword_overlap * 0.3 +
            category_match * 0.2
        ) / 1.5  # 归一化

        return min(semantic_similarity, 1.0)

    def find_similar_groups(self, qa_pairs: List[QAPair], similarity_threshold: float = 0.75) -> List[List[QAPair]]:
        """
        将相似的问答对分组

        Args:
            qa_pairs: 问答对列表
            similarity_threshold: 相似度阈值

        Returns:
            List[List[QAPair]]: 相似问答对分组
        """
        groups = []
        processed = set()

        for i, qa1 in enumerate(qa_pairs):
            if i in processed:
                continue

            current_group = [qa1]
            processed.add(i)

            for j, qa2 in enumerate(qa_pairs[i+1:], i+1):
                if j in processed:
                    continue

                similarity = self.calculate_semantic_similarity(qa1, qa2)
                if similarity >= similarity_threshold:
                    current_group.append(qa2)
                    processed.add(j)

            groups.append(current_group)

        # 过滤只有一个元素的组（不需要合并）
        similar_groups = [group for group in groups if len(group) > 1]
        single_groups = [group for group in groups if len(group) == 1]

        self.logger.info(f"相似性分析完成: {len(similar_groups)} 个需要合并的组, {len(single_groups)} 个独立问答对")

        return similar_groups + single_groups


class QACompactor:
    """智能问答对压缩器"""

    def __init__(self):
        """初始化压缩器"""
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

        # 相似性分析器
        self.similarity_analyzer = QASimilarityAnalyzer()

        # 压缩统计
        self.compaction_stats = {
            'total_compactions': 0,
            'total_qa_pairs_processed': 0,
            'total_merged_groups': 0,
            'total_duplicates_removed': 0,
            'compression_ratio': 0.0,
            'last_compaction_time': None
        }

    def get_merge_prompt(self) -> str:
        """
        获取问答对合并的prompt模板

        Returns:
            str: 合并prompt模板
        """
        prompt = """你是一个专业的知识库优化专家，负责合并博邦方舟无创血糖仪知识库中的相似问答对。

## 任务目标
将多个相似的问答对合并为一个高质量的问答对，确保信息完整、准确、无重复。

## 合并原则

### 核心要求
1. **信息完整性**：合并后的问答对应包含所有原始问答对的有用信息
2. **去重优化**：删除重复、冗余的信息
3. **质量提升**：优化语言表达，提高专业性和可读性
4. **逻辑清晰**：确保问答逻辑清晰，易于理解

### 合并策略
1. **问题合并**：
   - 选择最清晰、最全面的问题表述
   - 如有多个角度，可适当扩展问题范围
   - 保持问题的核心意图不变

2. **答案合并**：
   - 整合所有有价值的信息点
   - 按逻辑顺序组织答案结构
   - 删除重复或矛盾的内容
   - 优化语言表达，确保专业性

3. **元数据合并**：
   - 合并关键词，去除重复
   - 选择最准确的分类
   - 综合置信度评估
   - 保留最有价值的元数据

## 质量标准

### 问题质量
- 表述清晰、具体、易理解
- 符合用户实际咨询习惯
- 包含必要的上下文信息

### 答案质量
- 信息准确、完整、实用
- 逻辑结构清晰
- 语言专业、规范
- 适合知识库查询使用

### 整体质量
- 问答对应逻辑匹配
- 信息密度适中
- 便于检索和使用

## 输出格式

请严格按照以下JSON格式输出合并结果：

```json
{
    "merged_qa": {
        "question": "合并后的问题",
        "answer": "合并后的答案",
        "category": "问题分类",
        "keywords": ["关键词1", "关键词2", "关键词3"],
        "confidence": 0.95,
        "merge_notes": "合并说明"
    },
    "merge_analysis": {
        "original_count": 3,
        "merge_strategy": "信息整合策略说明",
        "improvements": ["改进点1", "改进点2"],
        "removed_duplicates": ["删除的重复信息"],
        "quality_score": 0.92
    }
}
```

### 字段说明
- **question**: 合并后的标准化问题
- **answer**: 合并后的完整答案
- **category**: 最合适的分类
- **keywords**: 合并后的关键词列表（3-8个）
- **confidence**: 合并结果的置信度（0.0-1.0）
- **merge_notes**: 合并过程的简要说明
- **merge_strategy**: 采用的合并策略说明
- **improvements**: 相对原问答对的改进点
- **removed_duplicates**: 删除的重复信息
- **quality_score**: 合并结果的质量评分（0.0-1.0）

## 示例

输入相似问答对：
```json
[
    {
        "question": "博邦方舟血糖仪怎么使用？",
        "answer": "使用很简单，开机后把手指放在测量区域就可以了",
        "category": "使用操作",
        "keywords": ["使用方法", "操作步骤"]
    },
    {
        "question": "无创血糖仪的操作步骤是什么？",
        "answer": "首先开机，然后将手指置于检测区，等待测量结果显示",
        "category": "使用操作",
        "keywords": ["操作步骤", "测量流程"]
    }
]
```

输出合并结果：
```json
{
    "merged_qa": {
        "question": "博邦方舟无创血糖仪的使用操作步骤是什么？",
        "answer": "博邦方舟无创血糖仪的使用非常简单：1. 首先开机启动设备；2. 将手指放置在测量检测区域；3. 等待设备自动测量并显示结果。整个过程无需采血，操作便捷。",
        "category": "使用操作",
        "keywords": ["使用方法", "操作步骤", "测量流程", "无创测量"],
        "confidence": 0.95,
        "merge_notes": "合并了两个相似的使用方法问答对，整合了操作步骤信息"
    },
    "merge_analysis": {
        "original_count": 2,
        "merge_strategy": "整合操作步骤，统一产品名称，优化表述逻辑",
        "improvements": ["统一了产品名称表述", "细化了操作步骤", "增加了无创特点说明"],
        "removed_duplicates": ["重复的开机和手指放置描述"],
        "quality_score": 0.92
    }
}
```

现在请对以下相似问答对进行合并：

"""
        return prompt

    def merge_similar_qa_pairs(self, similar_qa_pairs: List[QAPair]) -> Optional[QAPair]:
        """
        合并相似的问答对

        Args:
            similar_qa_pairs: 相似问答对列表

        Returns:
            QAPair: 合并后的问答对，失败返回None
        """
        if len(similar_qa_pairs) <= 1:
            return similar_qa_pairs[0] if similar_qa_pairs else None

        try:
            # 构建输入数据
            qa_data = []
            for qa in similar_qa_pairs:
                qa_data.append({
                    "question": qa.question,
                    "answer": qa.answer,
                    "category": qa.metadata.get('category', 'unknown'),
                    "keywords": qa.metadata.get('keywords', []),
                    "confidence": qa.metadata.get('confidence', 0.8),
                    "source_file": qa.source_file
                })

            # 构建完整的prompt
            full_prompt = self.get_merge_prompt() + "\n" + json.dumps(qa_data, ensure_ascii=False, indent=2)

            # 调用LLM API
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": full_prompt}
                ],
                temperature=0.1,
                max_tokens=3000,
            )

            result_text = response.choices[0].message.content.strip()

            # 解析合并结果
            merge_result = self._parse_merge_response(result_text)

            if merge_result:
                merged_qa_data = merge_result['merged_qa']

                # 创建合并后的QAPair
                merged_qa = QAPair(
                    id=str(uuid.uuid4()),
                    question=merged_qa_data['question'],
                    answer=merged_qa_data['answer'],
                    source_file=self._combine_source_files(similar_qa_pairs),
                    timestamp=datetime.now(),
                    metadata={
                        'category': merged_qa_data.get('category', 'unknown'),
                        'keywords': merged_qa_data.get('keywords', []),
                        'confidence': merged_qa_data.get('confidence', 0.8),
                        'merge_notes': merged_qa_data.get('merge_notes', ''),
                        'original_count': len(similar_qa_pairs),
                        'original_ids': [qa.id for qa in similar_qa_pairs],
                        'merge_analysis': merge_result.get('merge_analysis', {}),
                        'merge_time': datetime.now().isoformat(),
                        'merge_method': 'llm_intelligent'
                    }
                )

                self.logger.info(f"成功合并 {len(similar_qa_pairs)} 个相似问答对")
                return merged_qa

            else:
                self.logger.warning("LLM合并响应解析失败")
                return None

        except Exception as e:
            self.logger.error(f"合并相似问答对失败: {str(e)}")
            return None

    def _parse_merge_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        """
        解析LLM合并响应

        Args:
            response_text: LLM响应文本

        Returns:
            Dict[str, Any]: 解析后的合并结果，失败返回None
        """
        try:
            # 尝试提取JSON部分
            json_pattern = r'```json\n(.*?)\n```'
            json_match = re.search(json_pattern, response_text, re.DOTALL)

            if json_match:
                json_str = json_match.group(1)
            else:
                # 如果没有找到代码块，尝试查找JSON对象
                json_pattern = r'\{.*\}'
                json_match = re.search(json_pattern, response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    self.logger.warning("无法在响应中找到JSON格式数据")
                    return None

            # 解析JSON
            merge_data = json.loads(json_str)

            # 验证必要字段
            if 'merged_qa' not in merge_data:
                self.logger.warning("响应中缺少merged_qa字段")
                return None

            return merge_data

        except json.JSONDecodeError as e:
            self.logger.error(f"JSON解析失败: {str(e)}")
            return None
        except Exception as e:
            self.logger.error(f"响应解析失败: {str(e)}")
            return None

    def _combine_source_files(self, qa_pairs: List[QAPair]) -> str:
        """
        合并来源文件信息

        Args:
            qa_pairs: 问答对列表

        Returns:
            str: 合并后的来源文件信息
        """
        source_files = set()
        for qa in qa_pairs:
            source_files.add(qa.source_file)

        if len(source_files) == 1:
            return list(source_files)[0]
        else:
            return f"merged_from_{len(source_files)}_files"

    def detect_exact_duplicates(self, qa_pairs: List[QAPair]) -> Tuple[List[QAPair], List[QAPair]]:
        """
        检测并移除完全重复的问答对

        Args:
            qa_pairs: 问答对列表

        Returns:
            Tuple[List[QAPair], List[QAPair]]: (去重后的问答对, 重复的问答对)
        """
        seen_questions = {}
        unique_qa_pairs = []
        duplicate_qa_pairs = []

        for qa in qa_pairs:
            question_key = qa.question.strip().lower()

            if question_key in seen_questions:
                # 发现重复，比较答案质量
                existing_qa = seen_questions[question_key]
                existing_confidence = existing_qa.metadata.get('confidence', 0.8)
                current_confidence = qa.metadata.get('confidence', 0.8)

                if current_confidence > existing_confidence:
                    # 当前问答对质量更高，替换
                    unique_qa_pairs.remove(existing_qa)
                    duplicate_qa_pairs.append(existing_qa)
                    unique_qa_pairs.append(qa)
                    seen_questions[question_key] = qa
                else:
                    # 保持现有问答对，当前的标记为重复
                    duplicate_qa_pairs.append(qa)
            else:
                # 新问题，添加到唯一列表
                unique_qa_pairs.append(qa)
                seen_questions[question_key] = qa

        self.logger.info(f"去重完成: 保留 {len(unique_qa_pairs)} 个唯一问答对, 移除 {len(duplicate_qa_pairs)} 个重复")

        return unique_qa_pairs, duplicate_qa_pairs

    def compact_qa_pairs(self, qa_pairs: List[QAPair], similarity_threshold: float = 0.75) -> Dict[str, Any]:
        """
        压缩问答对列表

        Args:
            qa_pairs: 待压缩的问答对列表
            similarity_threshold: 相似度阈值

        Returns:
            Dict[str, Any]: 压缩结果
        """
        start_time = datetime.now()
        original_count = len(qa_pairs)

        self.logger.info(f"开始压缩 {original_count} 个问答对...")

        try:
            # 第一步：移除完全重复的问答对
            unique_qa_pairs, duplicates = self.detect_exact_duplicates(qa_pairs)
            self.logger.info(f"第一步完成：移除 {len(duplicates)} 个完全重复的问答对")

            # 第二步：相似性分析和分组
            similar_groups = self.similarity_analyzer.find_similar_groups(
                unique_qa_pairs, similarity_threshold
            )

            # 第三步：合并相似问答对
            compacted_qa_pairs = []
            merged_groups_count = 0
            total_tokens = 0

            for group in similar_groups:
                if len(group) > 1:
                    # 需要合并的组
                    merged_qa = self.merge_similar_qa_pairs(group)
                    if merged_qa:
                        compacted_qa_pairs.append(merged_qa)
                        merged_groups_count += 1
                        # 估算token使用（简化）
                        total_tokens += 1000  # 每次合并大约消耗1000 tokens
                    else:
                        # 合并失败，保留第一个
                        compacted_qa_pairs.append(group[0])
                        self.logger.warning(f"合并失败，保留第一个问答对: {group[0].question[:50]}...")
                else:
                    # 独立问答对，直接保留
                    compacted_qa_pairs.append(group[0])

            # 统计结果
            final_count = len(compacted_qa_pairs)
            compression_ratio = 1.0 - (final_count / original_count) if original_count > 0 else 0.0
            processing_time = (datetime.now() - start_time).total_seconds()

            # 更新统计信息
            self.compaction_stats.update({
                'total_compactions': self.compaction_stats['total_compactions'] + 1,
                'total_qa_pairs_processed': self.compaction_stats['total_qa_pairs_processed'] + original_count,
                'total_merged_groups': self.compaction_stats['total_merged_groups'] + merged_groups_count,
                'total_duplicates_removed': self.compaction_stats['total_duplicates_removed'] + len(duplicates),
                'compression_ratio': compression_ratio,
                'last_compaction_time': datetime.now().isoformat()
            })

            self.logger.info(f"压缩完成！")
            self.logger.info(f"原始数量: {original_count}, 压缩后: {final_count}")
            self.logger.info(f"压缩比例: {compression_ratio:.2%}")
            self.logger.info(f"处理时间: {processing_time:.1f}秒")

            return {
                'success': True,
                'original_count': original_count,
                'final_count': final_count,
                'compression_ratio': compression_ratio,
                'duplicates_removed': len(duplicates),
                'groups_merged': merged_groups_count,
                'processing_time': processing_time,
                'compacted_qa_pairs': compacted_qa_pairs,
                'estimated_tokens': total_tokens,
                'compaction_stats': self.compaction_stats.copy()
            }

        except Exception as e:
            self.logger.error(f"压缩过程失败: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'original_count': original_count,
                'processing_time': (datetime.now() - start_time).total_seconds()
            }

    def get_compaction_statistics(self) -> Dict[str, Any]:
        """
        获取压缩统计信息

        Returns:
            Dict[str, Any]: 压缩统计信息
        """
        return self.compaction_stats.copy()


class CompactionScheduler:
    """压缩调度器 - 管理定时压缩任务"""

    def __init__(self, compactor: QACompactor, interval_minutes: int = 5):
        """
        初始化压缩调度器

        Args:
            compactor: 压缩器实例
            interval_minutes: 压缩间隔（分钟）
        """
        self.logger = get_logger(__name__)
        self.compactor = compactor
        self.interval_minutes = interval_minutes
        self.is_running = False
        self.scheduler_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()

        # 压缩触发条件
        self.min_qa_pairs_threshold = 50  # 最少问答对数量才触发压缩
        self.compression_ratio_threshold = 0.1  # 最小压缩比例才有意义
        self.min_inactive_buffer_size = 20  # 非活跃缓冲区最小大小才压缩
        self.max_time_since_last_compaction = 60  # 最大时间间隔（分钟）

        # 压缩统计
        self.compaction_attempts = 0
        self.successful_compactions = 0
        self.failed_compactions = 0
        self.last_compaction_attempt = None
        self.last_successful_compaction = None

    def start_scheduler(self):
        """启动压缩调度器"""
        if self.is_running:
            self.logger.warning("压缩调度器已在运行")
            return

        self.is_running = True
        self.stop_event.clear()
        self.scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self.scheduler_thread.start()

        self.logger.info(f"压缩调度器启动，压缩间隔: {self.interval_minutes} 分钟")

    def stop_scheduler(self):
        """停止压缩调度器"""
        if not self.is_running:
            return

        self.is_running = False
        self.stop_event.set()

        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=30)

        self.logger.info("压缩调度器已停止")

    def _run_scheduler(self):
        """运行调度器循环"""
        while self.is_running and not self.stop_event.is_set():
            try:
                # 等待间隔时间
                if self.stop_event.wait(timeout=self.interval_minutes * 60):
                    # 收到停止信号
                    break

                # 执行压缩检查
                self._check_and_compact()

            except Exception as e:
                self.logger.error(f"调度器运行错误: {str(e)}")

    def _check_and_compact(self):
        """检查并执行压缩"""
        try:
            from src.core.knowledge_base import get_knowledge_base
            from datetime import datetime, timedelta

            self.compaction_attempts += 1
            self.last_compaction_attempt = datetime.now()

            self.logger.info("定时压缩检查中...")

            # 获取知识库实例
            knowledge_base = get_knowledge_base()

            # 获取知识库统计信息
            kb_stats = knowledge_base.get_statistics()
            total_qa_pairs = kb_stats.get('total_qa_pairs', 0)
            active_buffer_size = kb_stats.get('active_buffer_size', 0)
            inactive_buffer_size = kb_stats.get('inactive_buffer_size', 0)

            self.logger.debug(f"知识库状态: 总计{total_qa_pairs}个问答对, 活跃缓冲区{active_buffer_size}, 非活跃缓冲区{inactive_buffer_size}")

            # 检查是否满足压缩条件
            should_compact = False
            reasons = []

            # 条件1: 总问答对数量足够
            if total_qa_pairs < self.min_qa_pairs_threshold:
                self.logger.debug(f"问答对数量不足: {total_qa_pairs} < {self.min_qa_pairs_threshold}")
                return

            # 条件2: 活跃缓冲区有足够数据
            if active_buffer_size >= self.min_inactive_buffer_size:
                should_compact = True
                reasons.append(f"活跃缓冲区数据充足({active_buffer_size})")

            # 条件3: 检查距离上次压缩的时间
            compaction_stats = self.compactor.get_compaction_statistics()
            last_compaction_time_str = compaction_stats.get('last_compaction_time')

            if last_compaction_time_str:
                try:
                    last_compaction_time = datetime.fromisoformat(last_compaction_time_str)
                    time_since_last = (datetime.now() - last_compaction_time).total_seconds() / 60

                    if time_since_last >= self.max_time_since_last_compaction:
                        should_compact = True
                        reasons.append(f"距离上次压缩时间过长({time_since_last:.1f}分钟)")
                except:
                    # 如果解析时间失败，忽略此条件
                    pass
            else:
                # 从未压缩过，如果数据足够就压缩
                if active_buffer_size >= self.min_inactive_buffer_size:
                    should_compact = True
                    reasons.append("首次压缩")

            # 执行压缩
            if should_compact:
                self.logger.info(f"触发压缩: {', '.join(reasons)}")

                # 创建快照
                snapshot = knowledge_base.create_snapshot()
                if not snapshot:
                    self.logger.error("创建快照失败，跳过压缩")
                    self.failed_compactions += 1
                    return

                # 执行压缩
                compaction_result = self.compactor.compact_qa_pairs(
                    snapshot.data,
                    similarity_threshold=0.75
                )

                if compaction_result["success"]:
                    # 切换缓冲区并同步尾部数据
                    compacted_qa_pairs = compaction_result["compacted_qa_pairs"]
                    switch_success = knowledge_base.switch_buffers_with_tail_sync(compacted_qa_pairs)

                    if switch_success:
                        self.successful_compactions += 1
                        self.last_successful_compaction = datetime.now()

                        compression_ratio = compaction_result["compression_ratio"]
                        original_count = compaction_result["original_count"]
                        final_count = compaction_result["final_count"]

                        self.logger.info(f"✅ 定时压缩完成: {original_count} → {final_count} ({compression_ratio:.2%})")

                        # 如果压缩效果不明显，调整触发条件
                        if compression_ratio < self.compression_ratio_threshold:
                            self.min_inactive_buffer_size = min(self.min_inactive_buffer_size + 10, 100)
                            self.logger.info(f"压缩效果不明显，调整触发阈值至 {self.min_inactive_buffer_size}")

                    else:
                        self.failed_compactions += 1
                        self.logger.error("缓冲区切换失败")
                else:
                    self.failed_compactions += 1
                    self.logger.error(f"压缩失败: {compaction_result.get('error', '未知错误')}")

            else:
                self.logger.debug("暂不需要压缩")

        except Exception as e:
            self.failed_compactions += 1
            self.logger.error(f"定时压缩检查失败: {str(e)}")

    def get_scheduler_statistics(self) -> Dict[str, Any]:
        """获取调度器统计信息"""
        return {
            'is_running': self.is_running,
            'interval_minutes': self.interval_minutes,
            'compaction_attempts': self.compaction_attempts,
            'successful_compactions': self.successful_compactions,
            'failed_compactions': self.failed_compactions,
            'success_rate': self.successful_compactions / max(self.compaction_attempts, 1),
            'last_compaction_attempt': self.last_compaction_attempt.isoformat() if self.last_compaction_attempt else None,
            'last_successful_compaction': self.last_successful_compaction.isoformat() if self.last_successful_compaction else None,
            'trigger_thresholds': {
                'min_qa_pairs': self.min_qa_pairs_threshold,
                'min_inactive_buffer_size': self.min_inactive_buffer_size,
                'max_time_since_last_compaction_minutes': self.max_time_since_last_compaction,
                'compression_ratio_threshold': self.compression_ratio_threshold
            }
        }


# 全局压缩器实例
_compactor_instance: Optional[QACompactor] = None
_compactor_lock = threading.Lock()


def get_qa_compactor() -> QACompactor:
    """
    获取全局压缩器实例（单例模式）

    Returns:
        QACompactor: 压缩器实例
    """
    global _compactor_instance

    if _compactor_instance is None:
        with _compactor_lock:
            if _compactor_instance is None:
                _compactor_instance = QACompactor()

    return _compactor_instance