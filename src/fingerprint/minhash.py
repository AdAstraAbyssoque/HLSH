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
from typing import List, Set


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
        self.hash_functions = self._generate_hash_functions()

    def _generate_hash_functions(self) -> List[callable]:
        """
        生成一组哈希函数。

        返回:
            List[callable]: 哈希函数列表。
        """
        if self.seed is not None:
            random.seed(self.seed)

        hash_functions = []
        for _ in range(self.num_hashes):
            a = random.randint(1, 2**32 - 1)
            b = random.randint(0, 2**32 - 1)
            p = 2**33 - 355  # 一个大素数
            hash_functions.append(lambda x, a=a, b=b, p=p: (a * hash(x) + b) % p)
        return hash_functions

    def compute_signature(self, feature_set: Set[str]) -> List[int]:
        """
        计算 MinHash 签名。

        参数:
            feature_set (Set[str]): 输入特征集合（如 n-gram 或 token 集合）。

        返回:
            List[int]: 固定长度的 MinHash 签名。
        """
        signature = []
        for hash_func in self.hash_functions:
            min_hash = float("inf")
            for feature in feature_set:
                min_hash = min(min_hash, hash_func(feature))
            signature.append(min_hash)
        return signature

    def compare_signatures(self, sig1: List[int], sig2: List[int]) -> float:
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
        return sum(1 for i in range(len(sig1)) if sig1[i] == sig2[i]) / len(sig1)


# 示例用法
if __name__ == "__main__":
    # 示例特征集合
    feature_set1 = {"this", "is", "a", "test"}
    feature_set2 = {"this", "is", "another", "test","a"}

    # 初始化 MinHash
    minhash = MinHash(num_hashes=100, seed=42)

    # 计算签名
    signature1 = minhash.compute_signature(feature_set1)
    signature2 = minhash.compute_signature(feature_set2)

    # 比较签名相似性
    similarity = minhash.compare_signatures(signature1, signature2)
    print("MinHash 签名 1:", signature1)
    print("MinHash 签名 2:", signature2)
    print("MinHash 签名相似性:", similarity)