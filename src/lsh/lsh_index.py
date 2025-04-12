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
	•	调用：main.py在得到所有文档签名后，调用LSHIndex进行索引建立和候选对检索。
'''
from typing import List, Tuple, Set
from collections import defaultdict
import random
import itertools
from tqdm import tqdm


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
        for doc_id, signature in enumerate(signatures):
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
        for band in self.buckets:
            for bucket in band.values():
                if len(bucket) > 1:
                    for pair in itertools.combinations(bucket, 2):
                        candidate_pairs.add(tuple(sorted(pair)))
        return candidate_pairs


class SimHashLSHIndex:
    """
    基于 SimHash 签名的 LSH 索引，适用于 Hamming 距离。
    """

    def __init__(self, radius: int):
        """
        初始化 SimHash LSH 索引。
        参数:
            radius (int): 允许的 Hamming 距离误差半径。
        """
        self.radius = radius
        self.buckets = defaultdict(list)
        # 增加邻居缓存，减少重复生成
        self.neighbor_cache = {}

    def _generate_neighbors(self, signature: int) -> List[int]:
        """
        生成签名的所有可能近邻（通过 bit 翻转），只保留 Hamming 距离不超过 radius 的邻居。
        参数:
            signature (int): 输入签名。
        返回:
            List[int]: 近邻签名列表。
        """
        # 如果缓存存在则直接返回缓存结果
        
        if signature in self.neighbor_cache:
            return self.neighbor_cache[signature]

        neighbors = {signature}
        bits = 32  # 假设签名为 64 位
        # 对于距离 d = 1, 2, ..., self.radius 生成邻居
        for d in range(1, self.radius + 1):
            for bits_to_flip in itertools.combinations(range(bits), d):
                neighbor = signature
                for bit in bits_to_flip:
                    neighbor ^= (1 << bit)
                neighbors.add(neighbor)
        neighbor_list = list(neighbors)
        self.neighbor_cache[signature] = neighbor_list
        return neighbor_list

    def index(self, signatures: List[int]):
        """
        将签名存入哈希桶。
        参数:
            signatures (List[int]): SimHash 签名列表。
        """
        # 修改部分：为每个签名，不仅存储其自身键，还存储所有在允许半径内的邻居签名
        for doc_id, signature in tqdm(enumerate(signatures), desc="Indexing signatures", total=len(signatures)):
            neighbor_signatures = self._generate_neighbors(signature)
            for neighbor in neighbor_signatures:
                self.buckets[neighbor].append(doc_id)

    def query(self, signature: int) -> Set[int]:
        """
        查询可能相似的文档。
        参数:
            signature (int): 查询的 SimHash 签名。
        返回:
            Set[int]: 可能相似的文档 ID 集合。
        """
        # 如果需要查找与某个签名近似的文档，也可以直接生成邻居再查询，
        # 但此处由于 index 时已经将所有邻居都存入桶中，可以直接使用查询键
        return set(self.buckets.get(signature, []))

    def get_candidate_pairs(self) -> Set[Tuple[int, int]]:
        """
        返回所有候选文档对。
        返回:
            Set[Tuple[int, int]]: 候选文档对集合。
        """
        candidate_pairs = set()
        for bucket in tqdm(self.buckets.values(), desc="Processing buckets", total=len(self.buckets)):
            if len(bucket) > 1:
            # 桶中可能包含重复加入的同一个文档，这里可以利用 set 去重
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
            # 保证每个表的采样一致，可以指定随机种子
            rnd = random.Random(table_idx)
            sampled = rnd.sample(range(64), bits_per_table)
            self.sampled_bits.append(sampled)

    def _hash_signature(self, signature: int, table_idx: int) -> int:
        """
        对签名进行采样哈希，利用预采样的 bit 位。
        参数:
            signature (int): 输入签名。
            table_idx (int): 哈希表索引。
        返回:
            int: 哈希值。
        """
        sampled_bits = self.sampled_bits[table_idx]
        hash_value = 0
        for i, bit in enumerate(sampled_bits):
            if signature & (1 << bit):
                hash_value |= (1 << i)
        return hash_value

    def index(self, signatures: List[int]):
        """
        将签名存入多个采样哈希表。
        参数:
            signatures (List[int]): BitSampling 签名列表。
        """
        for doc_id, signature in tqdm(enumerate(signatures), desc="Indexing signatures", total=len(signatures)):
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



    def get_candidate_pairs(self) -> Set[Tuple[int, int]]:
        """
        返回所有候选文档对。
        返回:
            Set[Tuple[int, int]]: 候选文档对集合。
        """
        candidate_pairs = set()

        for table in tqdm(self.tables, desc="Processing tables", total=len(self.tables)):
            for bucket in table.values():
                if len(bucket) > 1:
                    for pair in itertools.combinations(bucket, 2):
                        candidate_pairs.add(tuple(sorted(pair)))
        return candidate_pairs


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
