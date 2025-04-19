'''
	•	功能：构建三种不同的 LSH 索引结构（MinHash、SimHash、BitSampling），用于根据签名快速生成候选近重复文档对。每种方法对应一个独立的索引类，分别适配不同的签名结构与相似度度量方式。
	•	主要类与函数：
	•	Class MinHashLSHIndex
		•	功能：基于 MinHash 签名，使用 banding 技术构建 LSH 索引，适用于 Jaccard 相似度。
		•	__init__(self, num_bands, rows_per_band)：初始化参数，包括 band 数量和每个 band 的哈希行数。
		•	index(self, signatures)：将签名分成多个 band，存入哈希桶。
		•	query(self, signature)：根据 band 匹配在桶中查找可能相似的文档。
		•	get_candidate_pairs(self)：返回所有桶中出现的候选文档对。
	•	Class SimHashLSHIndex
		•	功能：基于 SimHash 签名，使用 Hamming 距离范围查找近似匹配，适用于文本相似度。
		•	__init__(self, radius)：初始化允许的 Hamming 距离误差半径（bit 数差异）。
		•	index(self, signatures)：将每个签名映射到多个 Hamming 近邻桶（通过 bit flipping）。
		•	query(self, signature)：枚举签名的近邻桶，获取可能相似的文档。
		•	get_candidate_pairs(self)：综合所有桶中结果，输出候选文档对。
	•	Class BitSamplingLSHIndex
		•	功能：基于 BitSampling 签名，使用多个采样哈希表构建 LSH 索引，适用于 Hamming 相似度。
		•	__init__(self, num_hash_tables, bits_per_table)：初始化哈希表数量和每个哈希表使用的采样位数。
		•	index(self, signatures)：对每个文档签名执行多组位采样，将结果放入相应桶中。
		•	query(self, signature)：对给定签名按采样规则进行哈希，返回匹配桶中文档。
		•	get_candidate_pairs(self)：统计所有哈希表中桶的内容，生成候选文档对。
	•	Class HybridLSHIndex
		•	功能：基于 MinHash 和 SimHash 混合签名的 LSH 索引，支持多种合并策略。
		•	__init__(self, minhash_params, simhash_params, merge_strategy, weights)：初始化混合索引参数。
		•	index(self, signatures)：索引包含两种签名的文档数据。
		•	query(self, signature)：查询可能相似的文档。
		•	get_candidate_pairs(self)：根据合并策略获取候选文档对。
	•	调用：main.py在得到所有文档签名后，调用LSHIndex进行索引建立和候选对检索。
'''
from typing import List, Tuple, Set
from collections import defaultdict
import random
import itertools
from tqdm import tqdm
from multiprocessing import Pool


class MinHashLSHIndex:
    """
    基于 MinHash 签名的 LSH 索引，适用于 Jaccard 相似度。
    """

    def __init__(self, num_bands: int, rows_per_band: int):
        """
        初始化 MinHash LSH 索引。
        参数:
            num_bands (int): band 的数量。
            rows_per_band (int): 每个 band 的哈希行数。
        """
        self.num_bands = num_bands
        self.rows_per_band = rows_per_band
        self.buckets = [defaultdict(list) for _ in range(num_bands)]
        # 预先计算每个 band 的切片边界，格式为 (start, end)
        self.band_indices = [
            (band_idx * self.rows_per_band, (band_idx + 1) * self.rows_per_band)
            for band_idx in range(num_bands)
        ]

    def _hash_band(self, band: Tuple[int]) -> int:
        """
        对 band 进行哈希。
        参数:
            band (Tuple[int]): band 的内容。
        返回:
            int: 哈希值。
        """
        return hash(band)

    def index(self, signatures: List[List[int]]):
        """
        将签名分成多个 band 并存入哈希桶。
        参数:
            signatures (List[List[int]]): MinHash 签名列表。
        """
        for doc_id, signature in tqdm(enumerate(signatures), desc="Indexing signatures", total=len(signatures)):
            # 如果可能，可以提前将 signature 转换为元组，避免每个 band 都转换
            for band_idx, (start, end) in enumerate(self.band_indices):
                band = tuple(signature[start:end])
                bucket_key = self._hash_band(band)
                self.buckets[band_idx][bucket_key].append(doc_id)

    def query(self, signature: List[int]) -> Set[int]:
        """
        查询可能相似的文档。
        参数:
            signature (List[int]): 查询的 MinHash 签名。
        返回:
            Set[int]: 可能相似的文档 ID 集合。
        """
        candidates = set()
        for band_idx, (start, end) in enumerate(self.band_indices):
            band = tuple(signature[start:end])
            bucket_key = self._hash_band(band)
            candidates.update(self.buckets[band_idx].get(bucket_key, []))
        return candidates

    def get_candidate_pairs(self) -> Set[Tuple[int, int]]:
        """
        返回所有候选文档对。
        返回:
            Set[Tuple[int, int]]: 候选文档对集合。
        """
        candidate_pairs = set()
        for band in tqdm(self.buckets, desc="Processing bands", total=len(self.buckets)):
            for bucket in band.values():
                if len(bucket) > 1:
                    # 直接利用集合推导组合并更新候选对集合
                    candidate_pairs.update(
                        {tuple(sorted(pair)) for pair in itertools.combinations(bucket, 2)})
        return candidate_pairs


class SimHashLSHIndex:
    """
    LSH index for SimHash binary string signatures using segment-based bucketing.
    """

    def __init__(self, radius: int, hash_bits: int):
        """
        初始化 SimHash LSH 索引。
        :param radius: 最大 Hamming 距离（允许的误差半径）。
        :param hash_bits: 每个签名的二进制位数。
        """
        self.radius = radius
        self.hash_bits = hash_bits
        self.buckets: defaultdict[Tuple[int, str], List[int]] = defaultdict(list)
        self.signatures: List[str] = []

    @staticmethod
    def _hamming_distance(a: str, b: str) -> int:
        """
        计算两个等长二进制字符串的 Hamming 距离。
        :param a: 二进制字符串1。
        :param b: 二进制字符串2。
        :return: Hamming 距离。
        """
        return sum(ch1 != ch2 for ch1, ch2 in zip(a, b))

    def index(self, signatures: List[str]) -> None:
        """
        构建 LSH 索引，将每个签名分段存入桶中。
        :param signatures: 二进制字符串签名列表。
        """
        assert all(len(s) == self.hash_bits for s in signatures), \
            f"所有签名的长度必须为 {self.hash_bits} 位。"
        self.signatures = signatures
        k = self.radius + 1  # 分段数量
        part_len = self.hash_bits // k  # 每段的长度

        for idx, sig in tqdm(enumerate(signatures), desc="Indexing signatures", total=len(signatures)):
            for p in range(k):
                start = p * part_len
                end = (p + 1) * part_len if p < k - 1 else self.hash_bits
                seg = sig[start:end]
                key = (p, seg)
                self.buckets[key].append(idx)

    def query(self, sig: str) -> Set[int]:
        """
        查询与给定签名在 Hamming 距离范围内的文档。
        :param sig: 查询的二进制字符串签名。
        :return: 满足条件的文档索引集合。
        """
        assert len(sig) == self.hash_bits, f"查询签名的长度必须为 {self.hash_bits} 位。"
        k = self.radius + 1
        part_len = self.hash_bits // k
        candidates: Set[int] = set()

        for p in range(k):
            start = p * part_len
            end = (p + 1) * part_len if p < k - 1 else self.hash_bits
            seg = sig[start:end]
            key = (p, seg)
            candidates.update(self.buckets.get(key, []))

        # 精确过滤候选集合
        return {i for i in candidates if self._hamming_distance(self.signatures[i], sig) <= self.radius}

    def get_candidate_pairs(self) -> Set[Tuple[int, int]]:
        """
        生成所有满足 Hamming 距离条件的文档对。
        :return: 满足条件的文档对集合 (i, j)，其中 i < j。
        """
        pairs: Set[Tuple[int, int]] = set()
        for bucket in tqdm(self.buckets.values(), desc="Processing buckets", total=len(self.buckets)):
            unique_ids = sorted(set(bucket))
            if len(unique_ids) > 1:
                for i, j in itertools.combinations(unique_ids, 2):
                    if self._hamming_distance(self.signatures[i], self.signatures[j]) <= self.radius:
                        pairs.add((i, j))
        return pairs


class BitSamplingLSHIndex:
    """
    LSH index for SimHash binary‑string signatures using random‑bit sampling.

    * 预先为每个哈希表随机选取 bits_per_table 个比特位置；
    * 同一张表中，取签名在这些位置上的比特拼成 bucket key；
    * 查询或生成候选对时，只需在相同 key 的 bucket 内做 Hamming 距离过滤。
    """

    def __init__(
        self,
        radius: int,
        hash_bits: int,
        num_tables: int = 12,
        bits_per_table: int | None = None,
        seed: int | None = None,
    ):
        """
        :param radius: 最大 Hamming 距离阈值。
        :param hash_bits: 签名长度（bits）。
        :param num_tables: 哈希表（采样方案）数量，越大命中率越高、内存也越大。
        :param bits_per_table: 每张表采样的 bit 数；默认取 ceil(hash_bits / (radius + 1))。
        :param seed: 随机种子，便于结果复现。
        """
        self.radius = radius
        self.hash_bits = hash_bits
        self.num_tables = num_tables
        self.bits_per_table = bits_per_table or -(-hash_bits // (radius + 1))
        self.random = random.Random(seed)

        # 记录每张表采样到的 bit 位置，例如 [[3,17,44,…], …]
        self.tables_bits: List[List[int]] = [
            self.random.sample(range(hash_bits), self.bits_per_table)
            for _ in range(num_tables)
        ]

        # buckets[table_id][bucket_key] -> list[doc_id]
        self.buckets: List[defaultdict[str, List[int]]] = [
            defaultdict(list) for _ in range(num_tables)
        ]
        self.signatures: List[str] = []

    # ---------- 内部工具 ---------- #
    @staticmethod
    def _hamming_distance(a: str, b: str) -> int:
        return sum(ch1 != ch2 for ch1, ch2 in zip(a, b))

    def _bucket_key(self, sig: str, table_id: int) -> str:
        """拼接指定采样位上的比特，形成桶键。"""
        bits = self.tables_bits[table_id]
        return "".join(sig[p] for p in bits)

    # ---------- 构建索引 ---------- #
    def index(self, signatures: List[str]) -> None:
        """
        建立索引。
        :param signatures: 二进制字符串列表，长度均为 hash_bits。
        """
        assert all(len(s) == self.hash_bits for s in signatures), \
            f"All signatures must have length {self.hash_bits}"

        self.signatures = signatures

        for doc_id, sig in tqdm(
            enumerate(signatures), desc="Indexing signatures", total=len(signatures)
        ):
            for t in range(self.num_tables):
                key = self._bucket_key(sig, t)
                self.buckets[t][key].append(doc_id)

    # ---------- 单条查询 ---------- #
    def query(self, sig: str) -> Set[int]:
        """
        返回与查询签名 Hamming 距离 ≤ radius 的文档 id 集合。
        """
        assert len(sig) == self.hash_bits, \
            f"Query signature must have length {self.hash_bits}"

        candidates: Set[int] = set()
        for t in range(self.num_tables):
            key = self._bucket_key(sig, t)
            candidates.update(self.buckets[t].get(key, []))

        return {
            i for i in candidates
            if self._hamming_distance(self.signatures[i], sig) <= self.radius
        }

    # ---------- 批量候选对 ---------- #
    def get_candidate_pairs(self) -> Set[Tuple[int, int]]:
        """
        返回所有满足 Hamming 距离 ≤ radius 的文档对 (i, j)，且 i < j。
        """
        pairs: Set[Tuple[int, int]] = set()

        # 把每个表独立处理，避免重复遍历
        for t in range(self.num_tables):
            for bucket in tqdm(
                self.buckets[t].values(),
                desc=f"Processing buckets (table {t})",
                total=len(self.buckets[t]),
            ):
                uniq = sorted(set(bucket))
                if len(uniq) > 1:
                    for i, j in itertools.combinations(uniq, 2):
                        if self._hamming_distance(
                            self.signatures[i], self.signatures[j]
                        ) <= self.radius:
                            pairs.add((i, j))

        return pairs


class HybridLSHIndex:
    """
    基于 MinHash 和 SimHash 混合签名的 LSH 索引，支持多种合并策略。

    参数:
        minhash_params (dict): MinHash 参数，包含:
            num_bands (int): band 数量
            rows_per_band (int): 每个 band 的哈希行数
        simhash_params (dict): SimHash 参数，包含:
            radius (int): 允许的 Hamming 距离误差半径
            hash_bits (int): 签名位数(默认64)
        merge_strategy (str): 合并策略，可选:
            "union": 取两种方法的并集(默认)
            "intersection": 取两种方法的交集
            "two-stage": 先用simhash粗筛，再用minhash精筛
            "weighted": 加权合并
        weights (dict): 权重参数(仅weighted策略使用)，包含:
            "minhash" (float): MinHash权重
            "simhash" (float): SimHash权重
    """

    def __init__(self, minhash_params: dict, simhash_params: dict, merge_strategy: str = "union", weights: dict = None):
        """
        初始化混合 LSH 索引。
        参数:
            minhash_params (dict): MinHash 参数 {num_bands, rows_per_band}
            simhash_params (dict): SimHash 参数 {radius, hash_bits}
            merge_strategy (str): 合并策略，可选"union"/"intersection"/"two-stage"/"weighted"
            weights (dict): 权重参数 {"minhash": float, "simhash": float}，仅weighted策略使用
        """
        if weights is None:
            self.weights = {"minhash": 0.5, "simhash": 0.5}
        else:
            self.weights = weights

        self.minhash_lsh = MinHashLSHIndex(
            num_bands=minhash_params["num_bands"],
            rows_per_band=minhash_params["rows_per_band"]
        )
        self.simhash_lsh = SimHashLSHIndex(
            radius=simhash_params["radius"],
            hash_bits=simhash_params["hash_bits"]
        )
        self.minhash_signatures = []
        self.simhash_signatures = []
        self.merge_strategy = merge_strategy

    def index(self, signatures: List[List[int]]):
        """
        索引签名数据。
        参数:
            signatures (List[List[int]]): 包含[minhash_signature, simhash_signature]的列表
        """
        self.minhash_signatures = [sig[0] for sig in signatures]
        self.simhash_signatures = [sig[1] for sig in signatures]
        self.minhash_lsh.index(self.minhash_signatures)
        self.simhash_lsh.index(self.simhash_signatures)

    def query(self, signature: List[int]) -> Set[int]:
        """
        查询可能相似的文档。
        参数:
            signature (List[int]): [minhash_signature, simhash_signature]
        返回:
            Set[int]: 可能相似的文档 ID 集合
        """
        minhash_results = self.minhash_lsh.query(signature[0])
        simhash_results = self.simhash_lsh.query(signature[1])
        return minhash_results.union(simhash_results)

    def get_candidate_pairs(self) -> Set[Tuple[int, int]]:
        """
        获取混合方法的候选对。
        返回:
            Set[Tuple[int, int]]: 候选文档对集合
        """
        minhash_pairs = self.minhash_lsh.get_candidate_pairs()
        simhash_pairs = self.simhash_lsh.get_candidate_pairs()

        if self.merge_strategy == "union":
            return minhash_pairs.union(simhash_pairs)
        elif self.merge_strategy == "intersection":
            return minhash_pairs.intersection(simhash_pairs)
        elif self.merge_strategy == "two-stage":
            # 先用simhash粗筛，再用minhash精筛
            refined_pairs = set()
            for pair in simhash_pairs:
                doc1, doc2 = pair
                # 计算minhash相似度
                sig1 = self.minhash_signatures[doc1]
                sig2 = self.minhash_signatures[doc2]
                # 计算Jaccard相似度估计
                matching = sum(1 for a, b in zip(sig1, sig2) if a == b)
                jaccard_est = matching / len(sig1)
                # 如果Jaccard相似度高于阈值(使用rows_per_band作为参考)
                if jaccard_est >= (1.0 / self.minhash_lsh.rows_per_band):
                    refined_pairs.add(pair)
            return refined_pairs
        elif self.merge_strategy == "weighted":
            # 加权合并策略
            weighted_pairs = set()
            minhash_weight = self.weights["minhash"]
            simhash_weight = self.weights["simhash"]

            # 合并所有候选对
            all_pairs = minhash_pairs.union(simhash_pairs)

            for pair in all_pairs:
                doc1, doc2 = pair
                # 计算minhash相似度得分
                sig1 = self.minhash_signatures[doc1]
                sig2 = self.minhash_signatures[doc2]
                minhash_score = sum(1 for a, b in zip(
                    sig1, sig2) if a == b) / len(sig1)

                # 计算simhash相似度得分
                sig1 = self.simhash_signatures[doc1]
                sig2 = self.simhash_signatures[doc2]
                hamming_dist = bin(sig1 ^ sig2).count('1')
                simhash_score = 1 - (hamming_dist / self.simhash_lsh.bits)

                # 计算加权得分
                weighted_score = (minhash_score * minhash_weight +
                                  simhash_score * simhash_weight)

                # 如果加权得分超过阈值(0.5)
                if weighted_score >= 0.5:
                    weighted_pairs.add(pair)

            return weighted_pairs
        else:
            raise ValueError(f"未知的合并策略: {self.merge_strategy}")


# 示例用法
if __name__ == "__main__":
    # 示例签名
    minhash_signatures = [[1, 2, 3, 4, 5, 6], [1, 2, 3, 7, 8, 9]]
    simhash_signatures = [0b101010, 0b101011, 0b101011]
    bitsampling_signatures = [0b110011, 0b110010]

    # MinHash LSH 示例
    minhash_lsh = MinHashLSHIndex(num_bands=2, rows_per_band=3)
    minhash_lsh.index(minhash_signatures)
    print("MinHash 候选对:", minhash_lsh.get_candidate_pairs())
    print("MinHash 查询结果:", minhash_lsh.query([1, 2, 3, 4, 5, 6]))

    # SimHash LSH 示例
    simhash_lsh = SimHashLSHIndex(radius=1)
    simhash_lsh.index(simhash_signatures)
    print("SimHash 候选对:", simhash_lsh.get_candidate_pairs())
    print("SimHash 查询结果:", simhash_lsh.query(0b101010))

    # BitSampling LSH 示例
    bitsampling_lsh = BitSamplingLSHIndex(num_hash_tables=2, bits_per_table=3)
    bitsampling_lsh.index(bitsampling_signatures)
    print("BitSampling 候选对:", bitsampling_lsh.get_candidate_pairs())
    print("BitSampling 查询结果:", bitsampling_lsh.query(0b110011))
