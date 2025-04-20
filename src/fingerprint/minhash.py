'''
	•	功能：实现MinHash算法，生成MinHash签名并提供基于签名的相似性评估函数。
	•	主要类与函数：
	•	Class MinHash
	•	__init__(self, num_hashes, seed=None): 初始化时指定所使用的哈希函数数量及随机种子。
	•	compute_signature(self, feature_set): 输入如n-gram或token集合，输出固定长度的minhash签名。
	•	compare_signatures(self, sig1, sig2): 计算两个签名之间的近似Jaccard相似度。
	•	调用：在main.py中针对预处理后的特征调用MinHash生成签名。通常配合LSH中的banding技术使用。
'''
import random
import numpy as np
from typing import List, Set
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm


class MinHash:
    """
    MinHash 类，用于生成 MinHash 签名并计算签名之间的相似性。
    """

    def __init__(self, num_hashes: int, seed: int = None):
        """
        初始化 MinHash 实例。
        参数:
            num_hashes (int): 哈希函数的数量。
            seed (int): 随机种子，用于生成哈希函数（默认为 None）。
        """
        self.num_hashes = num_hashes
        self.seed = seed
        self.params = self._generate_hash_params()  # 存储(a, b, p)元组

    def _generate_hash_params(self) -> List[tuple]:
        """
        生成一组哈希函数参数，每个参数为 (a, b, p)。
        返回:
            List[tuple]: 哈希参数列表。
        """
        if self.seed is not None:
            random.seed(self.seed)
        params = []
        p = 2**33 - 355  # 一个大素数
        for _ in range(self.num_hashes):
            a = random.randint(1, 2**32 - 1)
            b = random.randint(0, 2**32 - 1)
            params.append((a, b, p))
        return params

    def compute_signature(self, feature_set: Set[str]) -> List[int]:
        """
        计算 MinHash 签名。
        参数:
            feature_set (Set[str]): 输入特征集合（如 n-gram 集合）。
        返回:
            List[int]: 固定长度的 MinHash 签名。
        """
        # 预计算每个 feature 的基础 hash 值，避免重复调用 hash(feature)
        hash_values = np.array([hash(feature) for feature in feature_set])
        signature = []

        # 利用 NumPy 向量化，对每个 (a, b, p) 参数批量计算所有 feature 的哈希值
        for a, b, p in self.params:
            # 计算所有 feature 的哈希值：(a * hash(feature) + b) mod p
            transformed = (a * hash_values + b) % p
            # 获取最小值作为当前 hash 函数的输出
            signature.append(int(transformed.min()))
        return signature

    def parallel_compute_signature(self, feature_sets: List[Set[str]],parallel_enable: bool, process_pool_size: int) -> List[List[int]]:
        """
        并行计算多个特征集合的 MinHash 签名。

        参数:
            feature_sets (List[Set[str]]): 输入的特征集合列表。
            process_pool_size (int): 并行处理的进程数。
            parallel_enable (bool): 是否启用并行处理。

        返回:
            List[List[int]]: 每个特征集合对应的 MinHash 签名列表。
        """
        signatures = []
        if not parallel_enable:
            signatures = [self.compute_signature(feature_set) for feature_set in tqdm(
                feature_sets, desc="生成 MinHash 签名")]
        else:
            # 使用 ProcessPoolExecutor 进行并行计算
            with ProcessPoolExecutor(max_workers=process_pool_size) as executor:
                signatures = list(tqdm(
                    executor.map(self.compute_signature, feature_sets),
                    total=len(feature_sets),
                    desc="并行生成 MinHash 签名"
                ))
        return signatures


def compare_signatures(sig1: List[int], sig2: List[int]) -> float:
    """
    比较两个 MinHash 签名的相似性。
    参数:
        sig1 (List[int]): 第一个 MinHash 签名。
        sig2 (List[int]): 第二个 MinHash 签名。
    返回:
        float: 近似 Jaccard 相似度。
    """
    if len(sig1) != len(sig2):
        raise ValueError("签名长度不一致，无法比较。")
    # 使用生成器表达式计算匹配率
    return sum(1 for i in range(len(sig1)) if sig1[i] == sig2[i]) / len(sig1)


# 示例用法
if __name__ == "__main__":
    # 示例特征集合
    feature_set1 = {"thi", "his", "is ", "s i", " is",
                    "is ", "s a", " a ", "a t", " te", "tes", "est"}
    feature_set2 = {"thi", "his", "is ", "s i", " is", "is ", "s a",
                    " a ", "a t", " te", "tes", "est", "ano", "not", "oth", "the", "her"}

    # 初始化 MinHash
    minhash = MinHash(num_hashes=100, seed=42)

    # 计算签名
    signature1 = minhash.compute_signature(feature_set1)
    signature2 = minhash.compute_signature(feature_set2)

    # 比较签名相似性
    similarity = compare_signatures(signature1, signature2)
    print("MinHash 签名 1:", signature1)
    print("MinHash 签名 2:", signature2)
    print("MinHash 签名相似性:", similarity)
