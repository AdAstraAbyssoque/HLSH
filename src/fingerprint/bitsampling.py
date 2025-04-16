'''
        •	功能：实现Bit Sampling技术。
        •	主要类与函数：
        •	Class BitSampling
        •	__init__(self, sample_size): 指定采样的位数或比例。
        •   vectorize(self, feature_set): 将特征集转换为二进制向量。
        •	compute_signature(self, feature_set): 根据输入特征采样生成二进制签名。
        •	调用：同样在main.py中，根据需要调用BitSampling进行实验与结果对比。
'''
from typing import Set, List
import random


from typing import Set, List
import random

class BitSampling:
    """
    实现 Bit Sampling 技术，用于生成二进制签名。
    支持可选的 TF-IDF 向量化（通过传入预先拟合好的 vectorizer），
    如果未传入，则采用简单 XOR 聚合。
    """

    def __init__(self, sample_size: int, hash_bits: int = 64, seed: int = None, vectorizer=None):
        """
        初始化 BitSampling 实例。

        参数:
            sample_size (int): 采样的位数。
            hash_bits (int): 输入签名的位数（默认为 64 位）。
            seed (int): 随机种子，用于生成采样位索引（默认为 None）。
            vectorizer: 可选的 TF-IDF 向量器（例如 TfidfVectorizer），需预先拟合语料库。
        """
        if sample_size > hash_bits:
            raise ValueError("采样位数不能大于输入签名的位数。")
        self.sample_size = sample_size
        self.hash_bits = hash_bits
        self.seed = seed
        self.vectorizer = vectorizer  # 可选 TF-IDF 向量器
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

    def vectorize(self, feature_set: Set[str]) -> int:
        """
        优化后的将特征集转换为二进制向量的方法：
        如果传入了 TF-IDF 向量器，则直接利用 vectorizer.vocabulary_ 和 idf_ 获取权重，
        避免每次调用 transform，从而提高效率。
        
        参数:
            feature_set (Set[str]): 输入特征集合。
        
        返回:
            int: 二进制向量表示。
        """
        if self.vectorizer is not None:
            weights = {}
            vocab = self.vectorizer.vocabulary_
            # idf_ 为 numpy array，索引对应 vocab 中的值
            idf = self.vectorizer.idf_
            for feature in feature_set:
                if feature in vocab:
                    weights[feature] = idf[vocab[feature]]
            return self.vectorize_weighted(feature_set, weights)
        else:
            # 简单 XOR 聚合
            vector = 0
            for feature in feature_set:
                hashed = hash(feature) & ((1 << self.hash_bits) - 1)
                vector ^= hashed
            return vector

    def vectorize_weighted(self, feature_set: Set[str], feature_weights: dict) -> int:
        """
        将特征集转换为加权后的二进制向量。
        对每个位进行累加权值计算，最后根据每个位的正负生成二进制表示，
        这里可以利用 TF-IDF 获得的权重。

        参数:
            feature_set (Set[str]): 输入特征集合。
            feature_weights (dict): 每个特征的权重字典。

        返回:
            int: 二进制向量表示。
        """
        score = [0] * self.hash_bits
        for feature in feature_set:
            weight = feature_weights.get(feature, 1)
            # 对特征哈希后，更新每个位的得分
            hashed = hash(feature)
            for i in range(self.hash_bits):
                bit = (hashed >> i) & 1
                score[i] += weight if bit else -weight
        vector = 0
        for i, val in enumerate(score):
            if val > 0:
                vector |= (1 << i)
        return vector

    def compute_signature(self, feature_set: Set[str], feature_weights: dict = None) -> int:
        """
        根据输入特征集合生成二进制签名。
        如果传入了 feature_weights，则使用加权向量化（例如TF-IDF），
        否则使用简单的 XOR 聚合。

        参数:
            feature_set (Set[str]): 输入特征集合。
            feature_weights (dict, 可选): 每个特征的权重字典。

        返回:
            int: 采样后的二进制签名。
        """
        if feature_weights is not None:
            full_vector = self.vectorize_weighted(feature_set, feature_weights)
        else:
            full_vector = self.vectorize(feature_set)
        signature = 0
        for i, bit_index in enumerate(self.sample_indices):
            if full_vector & (1 << bit_index):
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
        hamming_distance = bin(sig1 ^ sig2).count("1")
        return 1 - (hamming_distance / self.sample_size)


# 示例用法
if __name__ == "__main__":
    from sklearn.feature_extraction.text import TfidfVectorizer

    # 预先拟合一个 TF-IDF 向量器（示例中仅用当前样本文档拟合）
    corpus = ["this is a test", "this is another test"]
    vectorizer = TfidfVectorizer()
    vectorizer.fit(corpus)

    # 示例特征集合
    feature_set1 = {"this", "is", "a", "test"}
    feature_set2 = {"this", "is", "another", "test"}

    # 初始化 BitSampling，并传入 TF-IDF 向量器
    bitsampling = BitSampling(sample_size=16, hash_bits=64, seed=42, vectorizer=vectorizer)

    # 计算采样签名
    signature1 = bitsampling.compute_signature(feature_set1)
    signature2 = bitsampling.compute_signature(feature_set2)

    # 打印采样签名及其相似性
    print("采样签名 1:", bin(signature1))
    print("采样签名 2:", bin(signature2))
    print("采样签名相似性:", bitsampling.compare_signatures(signature1, signature2))