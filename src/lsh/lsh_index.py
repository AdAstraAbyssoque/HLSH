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
import math
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
                    for pair in itertools.combinations(bucket, 2):
                        candidate_pairs.add(tuple(sorted(pair)))
        return candidate_pairs


class SimHashLSHIndex:
    """
    基于 SimHash 签名的 LSH 索引，适用于 Hamming 距离。
    """

    def __init__(self, radius: int, hush_bits: int = 64, parallel_enabled: bool = False, process_pool_size: int = 4):
        self.radius = radius
        self.buckets = defaultdict(list)
        self.neighbor_cache = {}
        # 使用实际签名位数，建议与指纹生成时保持一致
        self.bits = hush_bits
        self.parallel_enabled = parallel_enabled
        self.process_pool_size = process_pool_size

    def _generate_neighbors_for_flip(self, args):
        signature, bits_to_flip = args
        neighbor = signature
        for bit in bits_to_flip:
            neighbor ^= (1 << bit)
        return neighbor

    def _generate_neighbors(self, signature: int) -> List[int]:
        """
        生成签名所有可能的近邻，保留 Hamming 距离不超过 radius 的邻居。
        """
        if signature in self.neighbor_cache:
            return self.neighbor_cache[signature]

        neighbors = {signature}
        max_d = min(self.radius, 2)  # 限制 radius 以防止任务数量爆炸
        tasks = []
        for d in range(1, max_d + 1):
            for bits_to_flip in itertools.combinations(range(self.bits), d):
                tasks.append((signature, bits_to_flip))

        if tasks:
            if self.parallel_enabled:
                with Pool(processes=self.process_pool_size) as pool:
                    results = pool.map(
                        self._generate_neighbors_for_flip, tasks)
            else:
                results = list(map(self._generate_neighbors_for_flip, tasks))
            for neighbor in results:
                neighbors.add(neighbor)
        neighbor_list = list(neighbors)
        self.neighbor_cache[signature] = neighbor_list
        return neighbor_list

    def index(self, signatures: List[int]):
        """
        将签名存入哈希桶。
        """
        for doc_id, signature in tqdm(enumerate(signatures), desc="Indexing signatures", total=len(signatures)):
            neighbor_signatures = self._generate_neighbors(signature)
            for neighbor in neighbor_signatures:
                self.buckets[neighbor].append(doc_id)

    def query(self, signature: int) -> Set[int]:
        return set(self.buckets.get(signature, []))

    def get_candidate_pairs(self) -> Set[Tuple[int, int]]:
        candidate_pairs = set()
        for bucket in tqdm(self.buckets.values(), desc="Processing buckets", total=len(self.buckets)):
            if len(bucket) > 1:
                # 使用 set() 移除重复文档 id，然后求组合
                for pair in itertools.combinations(set(bucket), 2):
                    candidate_pairs.add(tuple(sorted(pair)))
        return candidate_pairs


class BitSamplingLSHIndex:
    """
    基于 BitSampling 签名的 LSH 索引，适用于 Hamming 相似度。
    """

    def __init__(self, num_hash_tables: int, bits_per_table: int):
        """
        初始化 BitSampling LSH 索引。

        参数:
            num_hash_tables (int): 哈希表数量。
            bits_per_table (int): 每个哈希表使用的采样位数。
        """
        self.num_hash_tables = num_hash_tables
        self.bits_per_table = bits_per_table
        self.tables = [defaultdict(list) for _ in range(num_hash_tables)]
        # 预先采样每个哈希表需要的 bit 位
        self.sampled_bits = []
        for table_idx in range(num_hash_tables):
            rnd = random.Random(table_idx)  # 固定种子保证每个表一致
            sampled = rnd.sample(range(64), bits_per_table)
            self.sampled_bits.append(sampled)
        # 存储文档签名，便于候选对过滤
        self.doc_signatures = {}

    def _hash_signature(self, signature: int, table_idx: int) -> int:
        """
        对签名进行采样哈希，利用预采样的 bit 位。

        参数:
            signature (int): 输入签名。
            table_idx (int): 哈希表索引。

        返回:
            int: 哈希值。
        """
        hash_value = 0
        for i, bit in enumerate(self.sampled_bits[table_idx]):
            if signature & (1 << bit):
                hash_value |= (1 << i)
        return hash_value

    def index(self, signatures: List[int]):
        """
        将签名存入多个采样哈希表，并记录文档签名。

        参数:
            signatures (List[int]): BitSampling 签名列表。
        """
        for doc_id, signature in tqdm(enumerate(signatures), desc="Indexing signatures", total=len(signatures)):
            self.doc_signatures[doc_id] = signature
            for table_idx in range(self.num_hash_tables):
                bucket_key = self._hash_signature(signature, table_idx)
                self.tables[table_idx][bucket_key].append(doc_id)

    def query(self, signature: int) -> Set[int]:
        """
        查询可能相似的文档。

        参数:
            signature (int): 查询的 BitSampling 签名。

        返回:
            Set[int]: 可能相似的文档 ID 集合。
        """
        candidates = set()
        for table_idx in tqdm(range(self.num_hash_tables), desc="Querying tables", total=self.num_hash_tables):
            bucket_key = self._hash_signature(signature, table_idx)
            candidates.update(self.tables[table_idx].get(bucket_key, []))
        return candidates

    def get_candidate_pairs(self, min_similarity: float = 0.8, full_signature_bits: int = 64) -> Set[Tuple[int, int]]:
        """
        返回相似度高于阈值的候选文档对。

        参数:
            min_similarity (float): 签名相似度阈值，默认为 0.8。
            full_signature_bits (int): 完整签名的位数，用于构造位集合计算 Jaccard 相似度。

        返回:
            Set[Tuple[int, int]]: 候选文档对集合。
        """
        candidate_pairs = set()
        raw_pairs = set()
        for table in tqdm(self.tables, desc="Processing tables", total=len(self.tables)):
            for bucket in table.values():
                if len(bucket) > 1:
                    for pair in itertools.combinations(bucket, 2):
                        raw_pairs.add(tuple(sorted(pair)))
        # 使用完整签名计算 Jaccard 相似度
        for doc_id1, doc_id2 in tqdm(raw_pairs, desc="Filtering candidate pairs", total=len(raw_pairs)):
            sig1 = self.doc_signatures[doc_id1]
            sig2 = self.doc_signatures[doc_id2]
            # 将签名转换为位索引集合
            set1 = {i for i in range(full_signature_bits) if sig1 & (1 << i)}
            set2 = {i for i in range(full_signature_bits) if sig2 & (1 << i)}
            union = set1 | set2
            intersection = set1 & set2
            if union:
                similarity = len(intersection) / len(union)
            else:
                similarity = 1.0
            if similarity >= min_similarity:
                candidate_pairs.add((doc_id1, doc_id2))
        return candidate_pairs

class HybridLSHIndex:
    """
    基于 MinHash 和 SimHash 混合签名的 LSH 索引，支持多种合并策略。
    
    参数:
        minhash_params (dict): MinHash 参数，包含:
            num_bands (int): band 数量
            rows_per_band (int): 每个 band 的哈希行数
        simhash_params (dict): SimHash 参数，包含:
            radius (int): 允许的 Hamming 距离误差半径
            hush_bits (int): 签名位数(默认64)
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
            simhash_params (dict): SimHash 参数 {radius, hush_bits}
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
            hush_bits=simhash_params["hush_bits"]
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
                minhash_score = sum(1 for a, b in zip(sig1, sig2) if a == b) / len(sig1)
                
                # 计算simhash相似度得分
                sig1 = self.simhash_signatures[doc1]
                sig2 = self.simhash_signatures[doc2]
                hamming_dist = bin(sig1 ^ sig2).count('1')
                simhash_score = 1 - (hamming_dist / self.simhash_lsh.bits)
                
                # 计算加权得分
                weighted_score = math.sqrt(minhash_score * minhash_weight + simhash_score * simhash_weight)

                # 阈值
                if weighted_score >= 0.65 and minhash_score >= 0.3:
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
