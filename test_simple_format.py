#!/usr/bin/env python3
"""
测试简化后的LLM输出格式稳定性
"""

import os
import sys

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath('.'))

def test_simple_qa_parsing():
    """测试简化的QA对解析"""
    from src.core.qa_extractor import QAExtractor

    print("🔍 测试QA抽取器的简化格式解析...")

    extractor = QAExtractor()

    # 模拟LLM的简化输出
    simple_response = """Q: 博邦方舟无创血糖仪如何使用？
A: 使用方法很简单：1. 开机启动设备；2. 将手指放在检测区域；3. 等待测量结果显示。

Q: 设备的准确度如何？
A: 设备准确度很高，与传统血糖仪相比，误差控制在10%以内。

Q: 价格是多少？
A: 具体价格请咨询客服或查看官网信息。"""

    try:
        qa_pairs = extractor._parse_simple_qa_response(simple_response, "test_file.md")

        print(f"✅ 成功解析 {len(qa_pairs)} 个QA对")
        for i, qa in enumerate(qa_pairs, 1):
            print(f"  {i}. Q: {qa.question[:50]}...")
            print(f"     A: {qa.answer[:50]}...")

        return True
    except Exception as e:
        print(f"❌ QA解析失败: {e}")
        return False

def test_simple_merge_parsing():
    """测试简化的合并结果解析"""
    from src.core.qa_compactor import QACompactor
    from src.core.knowledge_base import QAPair
    from datetime import datetime
    import uuid

    print("\n🔍 测试压缩器的简化格式解析...")

    compactor = QACompactor()

    # 创建测试QA对
    test_qa_pairs = [
        QAPair(
            id=str(uuid.uuid4()),
            question="测试问题1",
            answer="测试答案1",
            source_file="test1.md",
            timestamp=datetime.now(),
            metadata={}
        ),
        QAPair(
            id=str(uuid.uuid4()),
            question="测试问题2",
            answer="测试答案2",
            source_file="test2.md",
            timestamp=datetime.now(),
            metadata={}
        )
    ]

    # 模拟LLM的简化合并输出
    simple_merge_response = """Q: 博邦方舟无创血糖仪的使用方法和准确度如何？
A: 博邦方舟无创血糖仪使用简单：开机后将手指放在检测区域即可。设备准确度很高，误差控制在10%以内，满足日常监测需求。"""

    try:
        merged_qa = compactor._parse_simple_merge_response(simple_merge_response, test_qa_pairs)

        if merged_qa:
            print(f"✅ 成功解析合并结果")
            print(f"  Q: {merged_qa.question}")
            print(f"  A: {merged_qa.answer[:100]}...")
            print(f"  原始QA对数量: {merged_qa.metadata.get('original_count', 0)}")
            return True
        else:
            print("❌ 合并解析返回None")
            return False
    except Exception as e:
        print(f"❌ 合并解析失败: {e}")
        return False

def test_simple_similarity_parsing():
    """测试简化的相似度分析解析"""
    from src.core.qa_compactor import QASimilarityAnalyzer
    from src.core.knowledge_base import QAPair
    from datetime import datetime
    import uuid

    print("\n🔍 测试相似度分析器的简化格式解析...")

    analyzer = QASimilarityAnalyzer()

    # 创建测试QA对
    test_qa_pairs = [
        QAPair(
            id=str(uuid.uuid4()),
            question="测试问题1",
            answer="测试答案1",
            source_file="test1.md",
            timestamp=datetime.now(),
            metadata={}
        ),
        QAPair(
            id=str(uuid.uuid4()),
            question="测试问题2",
            answer="测试答案2",
            source_file="test2.md",
            timestamp=datetime.now(),
            metadata={}
        ),
        QAPair(
            id=str(uuid.uuid4()),
            question="测试问题3",
            answer="测试答案3",
            source_file="test3.md",
            timestamp=datetime.now(),
            metadata={}
        )
    ]

    # 模拟LLM的简化相似度输出
    simple_similarity_response = """GROUP: 0,1
GROUP: 2"""

    try:
        similarity_result = analyzer._parse_simple_similarity_response(simple_similarity_response, test_qa_pairs)

        if similarity_result:
            analysis = similarity_result['group_analysis']
            print(f"✅ 成功解析相似度结果")
            print(f"  相似组数量: {len(analysis['similar_groups'])}")
            print(f"  独立QA数量: {len(analysis['independent_qa'])}")
            print(f"  总QA对数量: {analysis['analysis_summary']['total_qa_count']}")
            return True
        else:
            print("❌ 相似度解析返回None")
            return False
    except Exception as e:
        print(f"❌ 相似度解析失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 开始测试简化后的LLM输出格式稳定性\n")

    results = []

    # 测试QA抽取
    results.append(test_simple_qa_parsing())

    # 测试合并解析
    results.append(test_simple_merge_parsing())

    # 测试相似度解析
    results.append(test_simple_similarity_parsing())

    # 总结
    success_count = sum(results)
    total_count = len(results)

    print(f"\n📊 测试结果总结:")
    print(f"✅ 成功: {success_count}/{total_count}")
    print(f"❌ 失败: {total_count - success_count}/{total_count}")

    if success_count == total_count:
        print("🎉 所有测试通过！简化格式解析工作正常。")
        return True
    else:
        print("⚠️  部分测试失败，需要检查简化格式实现。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)