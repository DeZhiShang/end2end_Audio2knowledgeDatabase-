"""
Embedding相似度计算模块
使用qwen3-embedding-8b模型进行向量相似度计算和聚类预筛选
"""

import os
import json
import requests
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from sklearn.cluster import HDBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import threading
from dataclasses import dataclass
from datetime import datetime
import concurrent.futures
import time

from src.utils.logger import get_logger
from src.core.knowledge_base import QAPair
# 导入配置系统
from config import get_config


@dataclass
class EmbeddingCache:
    """向量缓存"""
    qa_id: str
    question_embedding: np.ndarray
    answer_embedding: np.ndarray
    combined_embedding: np.ndarray
    timestamp: datetime


class EmbeddingSimilarityCalculator:
    """基于Embedding的相似度计算器"""

    def __init__(self, embedding_url: str = "http://localhost:8001/v1/embeddings"):
        """
        初始化Embedding相似度计算器

        Args:
            embedding_url: qwen3-embedding-8b模型服务地址
        """
        self.logger = get_logger(__name__)
        self.embedding_url = embedding_url
        self.cache: Dict[str, EmbeddingCache] = {}
        self.cache_lock = threading.RLock()

        # 向量维度（qwen3-embedding-8b输出4096维）
        self.embedding_dim = 4096

        # 聚类参数 (从配置系统获取)
        self.min_cluster_size = get_config('algorithms.clustering.min_cluster_size', 2)
        self.min_samples = get_config('algorithms.clustering.min_samples', 2)

        # 相似度阈值 (从配置系统获取)
        self.high_similarity_threshold = get_config('models.embedding.high_similarity_threshold', 0.85)
        self.medium_similarity_threshold = get_config('models.embedding.medium_similarity_threshold', 0.75)
        self.low_similarity_threshold = get_config('models.embedding.low_similarity_threshold', 0.65)

        # 并行处理参数 (从配置系统获取)
        self.parallel_batch_size = get_config('models.embedding.parallel_batch_size', 35)
        self.max_workers = get_config('models.embedding.max_workers', 4)

        # Token预估 (从配置系统获取)
        self.avg_qa_tokens = get_config('algorithms.tokens.avg_qa_tokens', 200)
        self.embedding_context_limit = get_config('algorithms.tokens.embedding_context_limit', 16384)

    def get_embeddings_batch(self, texts: List[str], max_retries: int = 3) -> List[Optional[np.ndarray]]:
        """
        批量获取文本的向量表示

        Args:
            texts: 输入文本列表
            max_retries: 最大重试次数

        Returns:
            List[Optional[np.ndarray]]: 向量表示列表，失败的位置为None
        """
        if not texts:
            return []

        # 过滤空文本
        valid_texts = []
        text_indices = []
        for i, text in enumerate(texts):
            if text and text.strip():
                valid_texts.append(text.strip())
                text_indices.append(i)

        if not valid_texts:
            return [None] * len(texts)

        for attempt in range(max_retries):
            try:
                payload = {
                    "model": "qwen3-embedding-8b",
                    "input": valid_texts,  # 批量输入
                    "encoding_format": "float"
                }

                response = requests.post(
                    self.embedding_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=60  # 批量处理需要更长超时时间
                )

                if response.status_code == 200:
                    data = response.json()
                    if 'data' in data and len(data['data']) == len(valid_texts):
                        # 构建结果列表
                        results = [None] * len(texts)
                        for i, embedding_data in enumerate(data['data']):
                            embedding = np.array(embedding_data['embedding'], dtype=np.float32)
                            if embedding.shape[0] == self.embedding_dim:
                                results[text_indices[i]] = embedding
                            else:
                                self.logger.warning(f"向量维度不匹配: 期望{self.embedding_dim}, 实际{embedding.shape[0]}")
                        return results
                    else:
                        self.logger.error(f"批量嵌入响应格式错误: {data}")
                else:
                    self.logger.error(f"批量嵌入API调用失败: {response.status_code}, {response.text}")

            except requests.exceptions.Timeout:
                self.logger.warning(f"批量嵌入API超时，重试 {attempt + 1}/{max_retries}")
                if attempt == max_retries - 1:
                    self.logger.error("批量嵌入API多次超时，放弃请求")

            except Exception as e:
                self.logger.error(f"批量嵌入API调用异常: {str(e)}")
                break

        return [None] * len(texts)

    def get_embedding(self, text: str, max_retries: int = 3) -> Optional[np.ndarray]:
        """
        获取文本的向量表示

        Args:
            text: 输入文本
            max_retries: 最大重试次数

        Returns:
            np.ndarray: 向量表示，失败返回None
        """
        if not text or not text.strip():
            return None

        for attempt in range(max_retries):
            try:
                payload = {
                    "model": "qwen3-embedding-8b",
                    "input": text.strip(),
                    "encoding_format": "float"
                }

                response = requests.post(
                    self.embedding_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=30
                )

                if response.status_code == 200:
                    data = response.json()
                    if 'data' in data and len(data['data']) > 0:
                        embedding = np.array(data['data'][0]['embedding'], dtype=np.float32)
                        if embedding.shape[0] == self.embedding_dim:
                            return embedding
                        else:
                            self.logger.warning(f"向量维度不匹配: 期望{self.embedding_dim}, 实际{embedding.shape[0]}")
                    else:
                        self.logger.error(f"嵌入响应格式错误: {data}")
                else:
                    self.logger.error(f"嵌入API调用失败: {response.status_code}, {response.text}")

            except requests.exceptions.Timeout:
                self.logger.warning(f"嵌入API超时，重试 {attempt + 1}/{max_retries}")
                if attempt == max_retries - 1:
                    self.logger.error("嵌入API多次超时，放弃请求")

            except Exception as e:
                self.logger.error(f"嵌入API调用异常: {str(e)}")
                break

        return None

    def get_qa_embeddings_batch_parallel(self, qa_pairs: List[QAPair], batch_size: int = 35, max_workers: int = 4) -> Dict[str, EmbeddingCache]:
        """
        并行分批获取问答对的向量表示 - 解决长上下文问题

        Args:
            qa_pairs: 问答对列表
            batch_size: 每批次大小，避免embedding API上下文过长
            max_workers: 最大并行工作线程数

        Returns:
            Dict[str, EmbeddingCache]: QA ID到缓存向量数据的映射
        """
        results = {}
        uncached_pairs = []

        # 检查缓存
        with self.cache_lock:
            for qa in qa_pairs:
                if qa.id in self.cache:
                    results[qa.id] = self.cache[qa.id]
                else:
                    uncached_pairs.append(qa)

        if not uncached_pairs:
            return results

        self.logger.info(f"并行分批处理 {len(uncached_pairs)} 个未缓存的问答对，批次大小: {batch_size}，并行度: {max_workers}")

        # 将QA对分成多个小批次
        qa_batches = []
        for i in range(0, len(uncached_pairs), batch_size):
            batch = uncached_pairs[i:i + batch_size]
            qa_batches.append(batch)

        self.logger.info(f"分成 {len(qa_batches)} 个批次并行处理")

        # 并行处理所有批次
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有批次任务
            future_to_batch = {
                executor.submit(self._process_qa_batch, batch_idx, batch): batch_idx
                for batch_idx, batch in enumerate(qa_batches)
            }

            # 等待所有批次完成
            batch_results = {}
            for future in concurrent.futures.as_completed(future_to_batch):
                batch_idx = future_to_batch[future]
                try:
                    batch_result = future.result()
                    batch_results[batch_idx] = batch_result
                    self.logger.info(f"批次 {batch_idx + 1}/{len(qa_batches)} 完成，获得 {len(batch_result)} 个embedding")
                except Exception as e:
                    self.logger.error(f"批次 {batch_idx + 1} 处理失败: {str(e)}")
                    batch_results[batch_idx] = {}

        # 合并所有结果
        for batch_result in batch_results.values():
            results.update(batch_result)

        # 批量缓存结果
        with self.cache_lock:
            for qa_id, cache_entry in results.items():
                if qa_id not in self.cache:  # 只缓存新结果
                    self.cache[qa_id] = cache_entry

        self.logger.info(f"并行embedding处理完成，成功获得 {len([r for r in results.values() if r is not None])} 个向量")

        return results

    def _process_qa_batch(self, batch_idx: int, qa_batch: List[QAPair]) -> Dict[str, EmbeddingCache]:
        """
        处理单个QA批次 - 内部方法

        Args:
            batch_idx: 批次索引
            qa_batch: 问答对批次

        Returns:
            Dict[str, EmbeddingCache]: 批次结果
        """
        batch_results = {}

        try:
            # 准备批量文本：[question1, answer1, question2, answer2, ...]
            batch_texts = []
            for qa in qa_batch:
                batch_texts.append(qa.question)
                batch_texts.append(qa.answer)

            # 单次API调用获取所有embedding
            batch_embeddings = self.get_embeddings_batch(batch_texts)

            # 处理结果
            for i, qa in enumerate(qa_batch):
                question_idx = i * 2
                answer_idx = i * 2 + 1

                question_emb = batch_embeddings[question_idx] if question_idx < len(batch_embeddings) else None
                answer_emb = batch_embeddings[answer_idx] if answer_idx < len(batch_embeddings) else None

                if question_emb is not None and answer_emb is not None:
                    # 组合向量（加权平均：问题0.6，答案0.4）
                    combined_emb = 0.6 * question_emb + 0.4 * answer_emb
                    combined_emb = combined_emb / np.linalg.norm(combined_emb)  # 归一化

                    # 创建缓存条目
                    cache_entry = EmbeddingCache(
                        qa_id=qa.id,
                        question_embedding=question_emb,
                        answer_embedding=answer_emb,
                        combined_embedding=combined_emb,
                        timestamp=datetime.now()
                    )

                    batch_results[qa.id] = cache_entry
                else:
                    self.logger.warning(f"批次 {batch_idx + 1}: 无法获取QA对的embedding: {qa.question[:30]}...")

        except Exception as e:
            self.logger.error(f"批次 {batch_idx + 1} 处理异常: {str(e)}")

        return batch_results

    def get_qa_embeddings_batch(self, qa_pairs: List[QAPair], batch_size: int = 35) -> Dict[str, EmbeddingCache]:
        """
        批量获取问答对的向量表示（带缓存）

        Args:
            qa_pairs: 问答对列表
            batch_size: 批处理大小，避免单次请求过长

        Returns:
            Dict[str, EmbeddingCache]: QA ID到缓存向量数据的映射
        """
        results = {}
        uncached_pairs = []

        # 检查缓存
        with self.cache_lock:
            for qa in qa_pairs:
                if qa.id in self.cache:
                    results[qa.id] = self.cache[qa.id]
                else:
                    uncached_pairs.append(qa)

        if not uncached_pairs:
            return results

        self.logger.info(f"批量处理 {len(uncached_pairs)} 个未缓存的问答对")

        # 分批处理，避免超过embedding模型的上下文限制
        for batch_start in range(0, len(uncached_pairs), batch_size):
            batch_end = min(batch_start + batch_size, len(uncached_pairs))
            batch_qa_pairs = uncached_pairs[batch_start:batch_end]

            # 准备批量文本：[question1, answer1, question2, answer2, ...]
            batch_texts = []
            for qa in batch_qa_pairs:
                batch_texts.append(qa.question)
                batch_texts.append(qa.answer)

            # 批量获取embedding
            batch_embeddings = self.get_embeddings_batch(batch_texts)

            # 处理结果
            for i, qa in enumerate(batch_qa_pairs):
                question_idx = i * 2
                answer_idx = i * 2 + 1

                question_emb = batch_embeddings[question_idx] if question_idx < len(batch_embeddings) else None
                answer_emb = batch_embeddings[answer_idx] if answer_idx < len(batch_embeddings) else None

                if question_emb is not None and answer_emb is not None:
                    # 组合向量（加权平均：问题0.6，答案0.4）
                    combined_emb = 0.6 * question_emb + 0.4 * answer_emb
                    combined_emb = combined_emb / np.linalg.norm(combined_emb)  # 归一化

                    # 创建缓存条目
                    cache_entry = EmbeddingCache(
                        qa_id=qa.id,
                        question_embedding=question_emb,
                        answer_embedding=answer_emb,
                        combined_embedding=combined_emb,
                        timestamp=datetime.now()
                    )

                    # 缓存和返回结果
                    with self.cache_lock:
                        self.cache[qa.id] = cache_entry
                    results[qa.id] = cache_entry
                else:
                    self.logger.warning(f"无法获取QA对的embedding: {qa.question[:50]}...")

        return results

    def get_qa_embeddings(self, qa: QAPair) -> Optional[EmbeddingCache]:
        """
        获取问答对的向量表示（带缓存）

        Args:
            qa: 问答对

        Returns:
            EmbeddingCache: 缓存的向量数据，失败返回None
        """
        with self.cache_lock:
            # 检查缓存
            if qa.id in self.cache:
                return self.cache[qa.id]

        # 计算向量
        question_emb = self.get_embedding(qa.question)
        if question_emb is None:
            return None

        answer_emb = self.get_embedding(qa.answer)
        if answer_emb is None:
            return None

        # 组合向量（加权平均：问题0.6，答案0.4）
        combined_emb = 0.6 * question_emb + 0.4 * answer_emb
        combined_emb = combined_emb / np.linalg.norm(combined_emb)  # 归一化

        # 缓存结果
        cache_entry = EmbeddingCache(
            qa_id=qa.id,
            question_embedding=question_emb,
            answer_embedding=answer_emb,
            combined_embedding=combined_emb,
            timestamp=datetime.now()
        )

        with self.cache_lock:
            self.cache[qa.id] = cache_entry

        return cache_entry

    def calculate_similarity(self, qa1: QAPair, qa2: QAPair) -> Optional[float]:
        """
        计算两个问答对的相似度

        Args:
            qa1: 问答对1
            qa2: 问答对2

        Returns:
            float: 余弦相似度分数，失败返回None
        """
        emb1 = self.get_qa_embeddings(qa1)
        emb2 = self.get_qa_embeddings(qa2)

        if emb1 is None or emb2 is None:
            return None

        # 计算余弦相似度
        similarity = cosine_similarity(
            emb1.combined_embedding.reshape(1, -1),
            emb2.combined_embedding.reshape(1, -1)
        )[0][0]

        return float(similarity)

    def find_similar_clusters(self, qa_pairs: List[QAPair]) -> List[List[QAPair]]:
        """
        使用HDBSCAN聚类算法预筛选相似问答对

        Args:
            qa_pairs: 问答对列表

        Returns:
            List[List[QAPair]]: 聚类结果，每个子列表是一个相似组
        """
        if len(qa_pairs) < 2:
            return [qa_pairs] if qa_pairs else []

        # 智能计算批次大小，避免embedding上下文过长
        optimal_batch_size = min(
            self.parallel_batch_size,
            max(10, self.embedding_context_limit // (self.avg_qa_tokens * 2))  # *2 因为每个QA有问题+答案
        )

        # 并行分批获取所有向量 - 避免长上下文问题
        embedding_cache_map = self.get_qa_embeddings_batch_parallel(
            qa_pairs,
            batch_size=optimal_batch_size,
            max_workers=self.max_workers
        )

        embeddings = []
        valid_qa_pairs = []

        for qa in qa_pairs:
            if qa.id in embedding_cache_map:
                embeddings.append(embedding_cache_map[qa.id].combined_embedding)
                valid_qa_pairs.append(qa)
            else:
                self.logger.warning(f"无法获取向量，跳过QA: {qa.question[:50]}...")

        if len(embeddings) < 2:
            return [valid_qa_pairs] if valid_qa_pairs else []

        # HDBSCAN聚类
        embeddings_matrix = np.vstack(embeddings)

        try:
            clustering = HDBSCAN(
                min_cluster_size=self.min_cluster_size,
                min_samples=self.min_samples,
                metric='cosine'
            ).fit(embeddings_matrix)

            labels = clustering.labels_

            # 按标签分组
            clusters = {}
            for i, label in enumerate(labels):
                if label == -1:  # 噪声点，单独成组
                    clusters[f"noise_{i}"] = [valid_qa_pairs[i]]
                else:
                    if label not in clusters:
                        clusters[label] = []
                    clusters[label].append(valid_qa_pairs[i])

            cluster_list = list(clusters.values())

            # 过滤单个元素的簇（除非是噪声点）
            valid_clusters = []
            single_clusters = []

            for cluster in cluster_list:
                if len(cluster) > 1:
                    valid_clusters.append(cluster)
                else:
                    single_clusters.append(cluster)

            result = valid_clusters + single_clusters

            self.logger.info(f"HDBSCAN聚类完成: {len(embeddings)} → {len(valid_clusters)}个多元素簇 + {len(single_clusters)}个单元素簇")

            return result

        except Exception as e:
            self.logger.error(f"HDBSCAN聚类失败: {str(e)}")
            # 降级到单个QA对
            return [[qa] for qa in valid_qa_pairs]

    def rank_by_similarity(self, qa_pairs: List[QAPair], reference_qa: QAPair) -> List[Tuple[QAPair, float]]:
        """
        根据与参考QA对的相似度对列表排序

        Args:
            qa_pairs: 待排序的问答对列表
            reference_qa: 参考问答对

        Returns:
            List[Tuple[QAPair, float]]: 排序后的(问答对, 相似度)元组列表
        """
        similarities = []

        for qa in qa_pairs:
            if qa.id == reference_qa.id:
                continue

            similarity = self.calculate_similarity(reference_qa, qa)
            if similarity is not None:
                similarities.append((qa, similarity))
            else:
                # 无法计算相似度的放在最后
                similarities.append((qa, 0.0))

        # 按相似度降序排序
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities

    def get_cache_statistics(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        with self.cache_lock:
            return {
                'cached_embeddings': len(self.cache),
                'cache_memory_mb': len(self.cache) * self.embedding_dim * 3 * 4 / (1024 * 1024),  # 3个向量，4字节/float
                'embedding_dim': self.embedding_dim,
                'clustering_params': {
                    'min_cluster_size': self.min_cluster_size,
                    'min_samples': self.min_samples
                },
                'similarity_thresholds': {
                    'high': self.high_similarity_threshold,
                    'medium': self.medium_similarity_threshold,
                    'low': self.low_similarity_threshold
                }
            }

    def clear_cache(self):
        """清空向量缓存"""
        with self.cache_lock:
            cleared_count = len(self.cache)
            self.cache.clear()
            self.logger.info(f"已清空 {cleared_count} 个向量缓存")


class EmbeddingPrefilter:
    """基于Embedding的预筛选器"""

    def __init__(self, similarity_calculator: EmbeddingSimilarityCalculator):
        """
        初始化预筛选器

        Args:
            similarity_calculator: 相似度计算器
        """
        self.logger = get_logger(__name__)
        self.similarity_calc = similarity_calculator

    def prefilter_for_llm(self, qa_pairs: List[QAPair], batch_size: int = 35) -> List[List[QAPair]]:
        """
        为LLM分析预筛选相似问答对

        通过embedding聚类将大量问答对分成小批次，每个批次内的问答对有较高相似性概率
        这样LLM只需要在小范围内进行精确分析，大大降低token消耗

        Args:
            qa_pairs: 所有问答对
            batch_size: 每个批次的最大大小

        Returns:
            List[List[QAPair]]: 预筛选后的批次列表
        """
        if len(qa_pairs) <= batch_size:
            # 数量不多，直接返回单批次
            return [qa_pairs]

        self.logger.info(f"开始embedding预筛选: {len(qa_pairs)} 个问答对 → 目标批次大小 {batch_size}")

        # 第一步：HDBSCAN聚类预分组（使用并行embedding处理）
        clusters = self.similarity_calc.find_similar_clusters(qa_pairs)

        # 第二步：合并小簇，拆分大簇
        balanced_batches = []
        current_batch = []

        for cluster in clusters:
            if len(cluster) == 1:
                # 单个QA对，添加到当前批次
                current_batch.extend(cluster)

                if len(current_batch) >= batch_size:
                    balanced_batches.append(current_batch[:batch_size])
                    current_batch = current_batch[batch_size:]

            elif len(cluster) <= batch_size:
                # 中等大小簇，检查是否能加入当前批次
                if len(current_batch) + len(cluster) <= batch_size:
                    current_batch.extend(cluster)
                else:
                    # 当前批次满了，开始新批次
                    if current_batch:
                        balanced_batches.append(current_batch)
                    current_batch = cluster[:]

            else:
                # 大簇，需要拆分
                if current_batch:
                    balanced_batches.append(current_batch)
                    current_batch = []

                # 基于相似度排序后拆分
                cluster_sorted = self._sort_cluster_by_internal_similarity(cluster)

                for i in range(0, len(cluster_sorted), batch_size):
                    batch = cluster_sorted[i:i + batch_size]
                    balanced_batches.append(batch)

        # 处理剩余的问答对
        if current_batch:
            balanced_batches.append(current_batch)

        self.logger.info(f"预筛选完成: 生成 {len(balanced_batches)} 个批次，平均大小 {sum(len(b) for b in balanced_batches) / len(balanced_batches):.1f}")

        return balanced_batches

    def _sort_cluster_by_internal_similarity(self, cluster: List[QAPair]) -> List[QAPair]:
        """
        按内部相似度对簇进行排序
        将最相似的问答对放在一起，便于后续拆分时保持相似性

        Args:
            cluster: 簇中的问答对列表

        Returns:
            List[QAPair]: 排序后的问答对列表
        """
        if len(cluster) <= 2:
            return cluster

        # 选择第一个作为起点
        sorted_cluster = [cluster[0]]
        remaining = cluster[1:]

        # 贪心算法：每次选择与当前最后一个最相似的
        while remaining:
            current_qa = sorted_cluster[-1]
            similarities = self.similarity_calc.rank_by_similarity(remaining, current_qa)

            if similarities:
                next_qa, _ = similarities[0]  # 最相似的
                sorted_cluster.append(next_qa)
                remaining.remove(next_qa)
            else:
                # 无法计算相似度，随机选择一个
                sorted_cluster.append(remaining.pop(0))

        return sorted_cluster


# 全局实例
_embedding_calculator: Optional[EmbeddingSimilarityCalculator] = None
_embedding_calculator_lock = threading.Lock()


def get_embedding_similarity_calculator() -> EmbeddingSimilarityCalculator:
    """
    获取全局embedding相似度计算器实例（单例模式）

    Returns:
        EmbeddingSimilarityCalculator: 计算器实例
    """
    global _embedding_calculator

    if _embedding_calculator is None:
        with _embedding_calculator_lock:
            if _embedding_calculator is None:
                _embedding_calculator = EmbeddingSimilarityCalculator()

    return _embedding_calculator