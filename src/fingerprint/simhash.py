'''
	•	功能：实现SimHash算法。
	•	主要类与函数：
	•	Class SimHash
	•	__init__(self, hash_bits=64): 指定SimHash输出的位数（一般为64或128位）。
	•	compute_signature(self, feature_set): 对特征集合生成SimHash签名。
	•	hamming_distance(self, sig1, sig2): 计算两个SimHash签名的汉明距离，用于相似性比较。
	•	调用：main.py中可以根据配置选择SimHash方法进行指纹生成，并可与其他指纹方法进行对比分析。
'''

from typing import Set


class SimHash:
    """
    SimHash 类，用于生成 SimHash 签名并计算签名之间的汉明距离。
    """

    def __init__(self, hash_bits: int = 64):
        """
        初始化 SimHash 实例。

        参数:
            hash_bits (int): SimHash 输出的位数（默认为 64 位）。
        """
        self.hash_bits = hash_bits

    def _hash(self, token: str) -> int:
        """
        对单个 token 进行哈希，返回固定位数的整数。

        参数:
            token (str): 输入的特征 token。

        返回:
            int: 固定位数的哈希值。
        """
        h = hash(token) & ((1 << self.hash_bits) - 1)  # 截断为指定位数
        return h

    def compute_signature(self, feature_set: Set[str]) -> int:
        """
        计算 SimHash 签名。

        参数:
            feature_set (Set[str]): 输入特征集合。

        返回:
            int: SimHash 签名（整数表示）。
        """
        # 初始化权重向量
        vector = [0] * self.hash_bits

        for feature in feature_set:
            hashed = self._hash(feature)
            for i in range(self.hash_bits):
                # 根据哈希值的每一位更新权重向量
                if hashed & (1 << i):
                    vector[i] += 1
                else:
                    vector[i] -= 1

        # 根据权重向量生成签名
        signature = 0
        for i in range(self.hash_bits):
            if vector[i] > 0:
                signature |= (1 << i)

        return signature

    def hamming_distance(self, sig1: int, sig2: int) -> int:
        """
        计算两个 SimHash 签名的汉明距离。

        参数:
            sig1 (int): 第一个 SimHash 签名。
            sig2 (int): 第二个 SimHash 签名。

        返回:
            int: 汉明距离（不同位的数量）。
        """
        x = sig1 ^ sig2  # 异或操作，统计不同位
        return bin(x).count("1")


# 示例用法
if __name__ == "__main__":
    # 示例特征集合
    feature_set1 = {"this", "is", "a", "test"}
    feature_set2 = {"this", "is", "another", "test"}

    # 初始化 SimHash
    simhash = SimHash(hash_bits=64)

    # 计算签名
    signature1 = simhash.compute_signature(feature_set1)
    signature2 = simhash.compute_signature(feature_set2)

    # 打印签名
    print("SimHash 签名 1:", bin(signature1))
    print("SimHash 签名 2:", bin(signature2))

    # 计算汉明距离
    distance = simhash.hamming_distance(signature1, signature2)
    print("汉明距离:", distance)