"""
提示词模板管理系统

提供统一的提示词模板管理和渲染功能，避免硬编码，提高代码的可维护性和可扩展性。
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from enum import Enum


class PromptType(Enum):
    """提示词类型枚举"""
    CLEANING = "cleaning"
    GLEANING = "gleaning"
    QA_EXTRACTION = "qa_extraction"
    SIMILARITY = "similarity"
    MERGE = "merge"


class BasePromptTemplate(ABC):
    """提示词模板基类"""

    def __init__(self, template: str):
        """
        初始化提示词模板

        Args:
            template: 提示词模板字符串，可以包含 {key} 格式的占位符
        """
        self.template = template.strip()

    def render(self, **kwargs) -> str:
        """
        渲染提示词模板

        Args:
            **kwargs: 模板参数

        Returns:
            str: 渲染后的提示词
        """
        try:
            return self.template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"模板参数缺失: {e}")

    def __str__(self) -> str:
        return self.template


class StaticPromptTemplate(BasePromptTemplate):
    """静态提示词模板（不需要参数）"""

    def render(self, **kwargs) -> str:
        """静态模板直接返回模板内容"""
        return self.template


class ParameterizedPromptTemplate(BasePromptTemplate):
    """参数化提示词模板（需要参数）"""

    def __init__(self, template: str, required_params: Optional[list] = None):
        """
        初始化参数化模板

        Args:
            template: 模板字符串
            required_params: 必需参数列表
        """
        super().__init__(template)
        self.required_params = required_params or []

    def render(self, **kwargs) -> str:
        """
        渲染参数化模板

        Args:
            **kwargs: 模板参数

        Returns:
            str: 渲染后的提示词

        Raises:
            ValueError: 当缺少必需参数时
        """
        # 检查必需参数
        missing_params = [param for param in self.required_params if param not in kwargs]
        if missing_params:
            raise ValueError(f"缺少必需参数: {missing_params}")

        return super().render(**kwargs)


class PromptManager:
    """提示词管理器 - 统一管理所有提示词模板"""

    def __init__(self):
        """初始化提示词管理器"""
        self._templates: Dict[PromptType, BasePromptTemplate] = {}
        self._initialize_templates()

    def _initialize_templates(self):
        """初始化所有提示词模板"""
        # 数据清洗提示词
        self._templates[PromptType.CLEANING] = StaticPromptTemplate(self._get_cleaning_template())

        # 信息收集提示词 (需要轮数参数)
        self._templates[PromptType.GLEANING] = ParameterizedPromptTemplate(
            self._get_gleaning_template(),
            required_params=["round_number"]
        )

        # 问答抽取提示词
        self._templates[PromptType.QA_EXTRACTION] = StaticPromptTemplate(self._get_qa_extraction_template())

        # 相似度判断提示词
        self._templates[PromptType.SIMILARITY] = StaticPromptTemplate(self._get_similarity_template())

        # 合并提示词
        self._templates[PromptType.MERGE] = StaticPromptTemplate(self._get_merge_template())

    def get_prompt(self, prompt_type: PromptType, **kwargs) -> str:
        """
        获取指定类型的提示词

        Args:
            prompt_type: 提示词类型
            **kwargs: 模板参数

        Returns:
            str: 渲染后的提示词

        Raises:
            ValueError: 当提示词类型不存在或参数不正确时
        """
        if prompt_type not in self._templates:
            raise ValueError(f"不支持的提示词类型: {prompt_type}")

        template = self._templates[prompt_type]
        return template.render(**kwargs)

    def register_template(self, prompt_type: PromptType, template: BasePromptTemplate):
        """
        注册新的提示词模板

        Args:
            prompt_type: 提示词类型
            template: 提示词模板实例
        """
        self._templates[prompt_type] = template

    def list_templates(self) -> Dict[PromptType, str]:
        """
        列出所有已注册的提示词模板类型

        Returns:
            Dict[PromptType, str]: 模板类型和类名的映射
        """
        return {pt: type(template).__name__ for pt, template in self._templates.items()}

    # === 以下为各个提示词模板的具体内容 ===

    def _get_cleaning_template(self) -> str:
        """数据清洗提示词模板"""
        return """你是一个专业的语音识别数据清洗专家，负责清洗博邦方舟无创血糖仪客服对话记录。

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
   - 背景噪音（电视、音乐、其他人声）
   - 与血糖仪完全无关的对话
   - 过多的重复和口头禅

4. **标记内容**：
   - 不清楚的内容用"[不清楚]"标记
   - 推测的内容用"[可能是...]"标记

## 输出格式
请按照以下格式输出清洗后的对话：

```
# 博邦方舟无创血糖仪客服对话记录

**对话时间**: [如果有的话]
**参与人员**: 客服、用户

## 对话内容

**客服**: [清洗后的客服对话内容]
**用户**: [清洗后的用户对话内容]
...

## 清洗说明
[简要说明主要的清洗操作，比如修正了哪些识别错误，删除了哪些噪音内容]
```"""

    def _get_gleaning_template(self) -> str:
        """信息收集提示词模板"""
        return """你是一个专业的信息提取专家，负责从博邦方舟无创血糖仪客服对话中提取有价值的信息。

这是第 {round_number} 轮信息提取。请深入挖掘对话中可能遗漏的有价值信息。

## 任务目标
从已清洗的客服对话中提取更多有价值的信息，包括但不限于：
- 产品功能细节
- 用户使用场景
- 常见问题解决方案
- 产品优势介绍
- 技术参数信息

## 提取原则
1. **准确性**：基于对话实际内容，不编造信息
2. **全面性**：尽可能提取所有有价值的信息点
3. **结构化**：按照类别整理信息

## 输出格式
请按照以下格式输出提取的信息：

### 产品功能
- [具体功能描述]

### 使用方法
- [操作步骤或使用说明]

### 常见问题
- [问题及解决方案]

### 技术参数
- [相关技术指标或参数]

### 其他信息
- [其他有价值的信息]"""

    def _get_qa_extraction_template(self) -> str:
        """问答抽取提示词模板"""
        return """你是一个专业的知识库构建专家，专门从博邦方舟无创血糖仪客服对话中抽取高质量问答对。

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

## 注意事项
1. 一个对话可能包含多个问答对，请全部提取
2. 问题要站在用户角度，答案要体现客服专业性
3. 避免过于口语化的表达，适当规范化语言
4. 确保每个问答对都是独立完整的

## 过滤规则
1. 只抽取与博邦方舟无创血糖仪直接相关的问答（如功能、使用方法、价格、售后、适用人群、原理等）。
2. 严禁抽取任何与产品无关的信息，包括但不限于用户个人信息、闲聊内容、与其他产品相关的问题。
3. 如果对话中没有与产品相关的内容，则不要输出任何问答对，保持空白。

## 重中之中
对于没有任何知识库意义的对话，什么都不要返回，不要为了做而做。保持宁缺毋滥的原则
"""

    def _get_similarity_template(self) -> str:
        """相似度判断提示词模板"""
        return """你是一个专业的知识库内容分析专家，负责判断博邦方舟无创血糖仪知识库中问答对的相似性。

## 任务目标
分析问答对的相似性，判断它们是否可以合并。注意可能问题

## 相似性判断标准
- 高相似性（可合并）：
  - 问题表达方式不同，但核心意图相同
  - 答案内容相同或仅在表述、细节程度上不同（如一个更简洁，一个更详细）
- 低相似性（不可合并）：
  - 问题意图不同
  - 答案包含差异性信息（如数值不同、条件不同、场景不同）

遇到边界情况时：
- 如果答案差异不影响核心含义 → 合并
- 如果答案差异可能导致用户得到不同结论 → 不合并

## 输出格式
- 只输出分组结果，每个相似QA对组一行
- 格式：
  GROUP: QA编号1,QA编号2,QA编号3
- 独立QA对不输出任何内容
- 不要输出解释或分析

## 注意事项
- 给出的QA列可能有多个相似GROUP，注意甄别，别只合并成一个GROUP。多个相似组使用不同的行输出
- 示例仅仅是给你一个简要的示例，并不代表其中信息是正确的，一切的信息来源都基于实际提供的问答对

## 示例

输入QA对：
0. Q: 博邦方舟血糖仪怎么使用？
   A: 使用很简单，开机后把手指放在测量区域就可以了

1. Q: 无创血糖仪的操作步骤是什么？
   A: 首先开机，然后将手指置于检测区，等待测量结果显示

2. Q: 设备的价格是多少？
   A: 具体价格请咨询客服或查看官网信息

3. Q：你们这东西多少钱呀
   A：个人版2199元，家庭版2999元

输出：
GROUP: 0,1
GROUP：2,3
"""

    def _get_merge_template(self) -> str:
        """合并提示词模板"""
        return """你是一个专业的知识库内容合并专家，负责将相似的博邦方舟无创血糖仪问答对合并成高质量的统一问答对。

## 任务目标
将相似的问答对合并成一个更完整、更专业的问答对。

## 合并原则

### 问题合并
1. **保持核心意图**：选择最清晰、最常见的问法作为主问题
2. **语言规范化**：使用标准、专业的表达方式
3. **避免冗余**：去除不必要的口语化表达

### 答案合并
1. **信息完整性**：整合所有有价值的信息点
2. **逻辑清晰**：按照逻辑顺序组织答案内容
3. **专业准确**：确保技术信息的准确性
4. **详略得当**：既要详细又要重点突出

### 质量要求
1. **准确性**：不能改变原有信息的真实性
2. **完整性**：包含所有相似问答对的有效信息
3. **专业性**：体现客服专业水准
4. **实用性**：对用户具有实际帮助价值

## 输出格式
直接输出合并后的问答对，格式如下：

Q: [合并后的问题]
A: [合并后的答案]

## 注意事项
1. 如果答案中有冲突信息，选择更准确或更详细的版本
2. 保持答案的逻辑性和可读性
3. 避免信息重复，但确保完整性
4. 使用规范的语言表达，避免过于口语化"""


# 全局提示词管理器实例
prompt_manager = PromptManager()


def get_prompt(prompt_type: PromptType, **kwargs) -> str:
    """
    便捷函数：获取指定类型的提示词

    Args:
        prompt_type: 提示词类型
        **kwargs: 模板参数

    Returns:
        str: 渲染后的提示词
    """
    return prompt_manager.get_prompt(prompt_type, **kwargs)


# 为了向后兼容，提供便捷函数
def get_cleaning_prompt() -> str:
    """获取数据清洗提示词"""
    return get_prompt(PromptType.CLEANING)


def get_gleaning_prompt(round_number: int) -> str:
    """获取信息收集提示词"""
    return get_prompt(PromptType.GLEANING, round_number=round_number)


def get_qa_extraction_prompt() -> str:
    """获取问答抽取提示词"""
    return get_prompt(PromptType.QA_EXTRACTION)


def get_similarity_prompt() -> str:
    """获取相似度判断提示词"""
    return get_prompt(PromptType.SIMILARITY)


def get_merge_prompt() -> str:
    """获取合并提示词"""
    return get_prompt(PromptType.MERGE)