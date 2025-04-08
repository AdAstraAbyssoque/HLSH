'''
        •	功能：实现Bit Sampling技术。
        •	主要类与函数：
        •	Class BitSampling
        •	__init__(self, sample_size): 指定采样的位数或比例。
        •	compute_signature(self, feature_set): 根据输入特征采样生成二进制签名。
        •	调用：同样在main.py中，根据需要调用BitSampling进行实验与结果对比。
'''

from typing import List, Set
import random


class BitSampling:
    """
    BitSampling 类，用于实现 Bit Sampling 技术。
    """

    def __init__(self, sample_size: int, hash_bits: int = 64, seed: int = None):
        """
        初始化 BitSampling 实例。

        参数:
            sample_size (int): 采样的位数。
            hash_bits (int): 输入签名的位数（默认为 64 位）。
            seed (int): 随机种子，用于生成采样位索引（默认为 None）。
        """
        if sample_size > hash_bits:
            raise ValueError("采样位数不能大于输入签名的位数。")
        self.sample_size = sample_size
        self.hash_bits = hash_bits
        self.seed = seed
        self.sample_indices = self._generate_sample_indices()

    def _generate_sample_indices(self) -> List[int]:
        """
        生成采样位的索引。

        返回:
            List[int]: 采样位的索引列表。
        """
        if self.seed is not None:
            random.seed(self.seed)
        return random.sample(range(self.hash_bits), self.sample_size)

    def compute_signature(self, feature_set: Set[str]) -> int:
        """
        根据输入特征集合生成二进制签名。

        参数:
            feature_set (Set[str]): 输入特征集合。

        返回:
            int: 采样后的二进制签名。
        """
        # 计算完整的 SimHash 签名
        full_signature = self._compute_full_signature(feature_set)

        # 根据采样位生成采样签名
        sampled_signature = 0
        for i, bit_index in enumerate(self.sample_indices):
            if full_signature & (1 << bit_index):
                sampled_signature |= (1 << i)

        return sampled_signature

    def _compute_full_signature(self, feature_set: Set[str]) -> int:
        """
        计算完整的 SimHash 签名（辅助方法）。

        参数:
            feature_set (Set[str]): 输入特征集合。

        返回:
            int: 完整的 SimHash 签名。
        """
        vector = [0] * self.hash_bits

        for feature in feature_set:
            hashed = hash(feature) & ((1 << self.hash_bits) - 1)  # 截断为指定位数
            for i in range(self.hash_bits):
                if hashed & (1 << i):
                    vector[i] += 1
                else:
                    vector[i] -= 1

        signature = 0
        for i in range(self.hash_bits):
            if vector[i] > 0:
                signature |= (1 << i)

        return signature

    def compare_signatures(self, sig1: int, sig2: int) -> float:
        """
        比较两个采样签名的相似性。

        参数:
            sig1 (int): 第一个采样签名。
            sig2 (int): 第二个采样签名。

        返回:
            float: 相似性度量（0 到 1 之间）。
        """
        # 计算汉明距离
        hamming_distance = bin(sig1 ^ sig2).count("1")
        return 1 - (hamming_distance / self.sample_size)


# 示例用法
if __name__ == "__main__":
    # 示例特征集合
    feature_set1 = {"this", "is", "a", "test"}
    feature_set2 = {"this", "is", "another", "test"}

    # 初始化 BitSampling
    bitsampling = BitSampling(sample_size=16, hash_bits=64, seed=42)

    # 计算采样签名
    signature1 = bitsampling.compute_signature(feature_set1)
    signature2 = bitsampling.compute_signature(feature_set2)

    # 打印采样签名
    print("采样签名 1:", bin(signature1))
    print("采样签名 2:", bin(signature2))
    print("采样签名相似性:", bitsampling.compare_signatures(signature1, signature2))
