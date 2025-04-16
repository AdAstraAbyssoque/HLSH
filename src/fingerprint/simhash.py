'''
	•	功能：实现SimHash算法。
	•	主要类与函数：
	•	Class SimHash
	•	__init__(self, hash_bits=64): 指定SimHash输出的位数（一般为64或128位）。
	•	_compute_frequency(self, frequency:list[dict(str:int)]): 对特征集合生成SimHash签名。
    •	_compute_tfidf(self,tokens: list[set[str]],idf) : 对特征集合生成SimHash签名。
    •	compute_signature(self, feature_set,idf=none): 输入特征集合，输出SimHash签名（二进制表示）。
	•	hamming_distance(self, sig1, sig2): 计算两个SimHash签名的汉明距离，用于相似性比较。
	•	调用：main.py中可以根据配置选择SimHash方法进行指纹生成，并可与其他指纹方法进行对比分析。
'''
import math
from collections import Counter
from typing import List, Set
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm


class SimHash:
    """
    SimHash 类，实现了基于词频或 TF-IDF 权重的 SimHash 签名生成，
    以及基于汉明距离的签名比较，输出为二进制字符串。
    """

    def __init__(self, hash_bits: int = 64):
        """
        初始化 SimHash 实例，指定输出签名的位数。

        参数:
            hash_bits (int): SimHash 输出的签名位数（一般为64或128位）。
        """
        self.hash_bits = hash_bits

    def _compute_frequency(self, frequency: dict) -> int:
        """
        基于词频信息计算 SimHash 签名。

        参数:
            frequency (dict): 键为单词，值为该单词在文本中的频率。

        返回:
            int: 计算得到的 SimHash 签名（整数表示）。
        """
        bit_weights = [0] * self.hash_bits
        for token, weight in frequency.items():
            token_hash = hash(token) & ((1 << self.hash_bits) - 1)
            for i in range(self.hash_bits):
                if token_hash & (1 << i):
                    bit_weights[i] += weight
                else:
                    bit_weights[i] -= weight

        signature = 0
        for i in range(self.hash_bits):
            if bit_weights[i] > 0:
                signature |= (1 << i)
        return signature

    def _compute_tfidf(self, feature_set: set, idf: dict) -> int:
        """
        基于 TF-IDF 权重计算 SimHash 签名。
        对于每个 token，如果在 idf 中存在，则使用相应的 idf 权重；否则默认权重为1。

        参数:
            feature_set (set): 输入特征集合（例如 token 或 n-gram 集合）。
            idf (dict): 每个 token 对应的 idf 值，形式为 {token: idf_value}。

        返回:
            int: 计算得到的 SimHash 签名（整数表示）。
        """
        frequency = {token: idf.get(token, 1) for token in feature_set}
        return self._compute_frequency(frequency)

    def compute_signature(self, feature_set: set, idf: dict = None) -> str:
        """
        根据输入的特征集合生成 SimHash 签名。如果提供 idf 权重信息，则基于 TF-IDF 计算；
        否则基于词频（所有 token 权重置为1）。
        输出为二进制字符串，宽度为 hash_bits 位。

        参数:
            feature_set (set): 输入特征集合。
            idf (dict, 可选): 每个 token 对应的 idf 值。

        返回:
            str: SimHash 签名（二进制字符串）。
        """
        if idf is not None:
            sig = self._compute_tfidf(feature_set, idf)
        else:
            frequency = {token: 1 for token in feature_set}
            sig = self._compute_frequency(frequency)
        return format(sig, '0{}b'.format(self.hash_bits))

    def parallel_compute_signature(self, feature_sets: List[Set[str]], idf: dict = None, parallel_enable: bool = False, process_pool_size: int = 4) -> List[str]:
        """
        并行计算多个特征集合的 SimHash 签名，输出均为二进制字符串。

        参数:
            feature_sets (List[Set[str]]): 输入特征集合列表。
            idf (dict, 可选): 每个 token 对应的 idf 值。
            parallel_enable (bool): 是否启用并行计算。
            process_pool_size (int): 进程池大小。

        返回:
            List[str]: 每个特征集合对应的 SimHash 签名（以二进制字符串形式返回）。
        """
        signatures = []
        if parallel_enable:
            with ProcessPoolExecutor(max_workers=process_pool_size) as executor:
                futures = []
                for feature_set in tqdm(feature_sets, desc="提交任务到进程池"):
                    if idf is not None:
                        futures.append(executor.submit(self._compute_tfidf, feature_set, idf))
                    else:
                        futures.append(executor.submit(self._compute_frequency, {token: 1 for token in feature_set}))
                for future in tqdm(futures, desc="并行计算 SimHash 签名"):
                    int_sig = future.result()
                    signatures.append(format(int_sig, '0{}b'.format(self.hash_bits)))
        else:
            for feature_set in tqdm(feature_sets, desc="计算 SimHash 签名"):
                signatures.append(self.compute_signature(feature_set, idf))
        return signatures


def hamming_distance(sig1: str, sig2: str) -> int:
    """
    计算两个 SimHash 签名（二进制字符串）的汉明距离，
    即对应位上不同的数量。

    参数:
        sig1 (str): 第一个签名（二进制字符串）。
        sig2 (str): 第二个签名（二进制字符串）。

    返回:
        int: 两个签名之间的汉明距离。
    """
    # 先将二进制字符串转换为整数
    int1 = int(sig1, 2)
    int2 = int(sig2, 2)
    x = int1 ^ int2
    if hasattr(x, "bit_count"):
        return x.bit_count()
    distance = 0
    while x:
        distance += 1
        x &= (x - 1)
    return distance


# 示例用法
if __name__ == "__main__":
    text = "This is a test, this test is simple and effective."
    tokens = set(text.lower().replace(',', '').replace('.', '').split())

    simhash = SimHash(hash_bits=64)

    # 示例1：基于词频
    binary_sig1 = simhash.compute_signature(tokens)
    print("SimHash 签名（基于词频）：", binary_sig1)

    # 示例2：基于 TF-IDF 权重
    idf = {token: 1.5 for token in tokens}
    binary_sig2 = simhash.compute_signature(tokens, idf=idf)
    print("SimHash 签名（基于 TF-IDF）：", binary_sig2)

    distance = hamming_distance(binary_sig1, binary_sig2)
    print("汉明距离：", distance)