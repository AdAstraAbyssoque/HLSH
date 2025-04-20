'''
    • Functionality: Constructs three different types of LSH index structures (MinHash, SimHash, BitSampling) for quickly generating candidate near-duplicate document pairs based on signatures. Each method corresponds to a separate index class, adapted to different signature structures and similarity measures.
    • Main classes and functions:
    • Class MinHashLSHIndex
        • Functionality: Builds an LSH index using MinHash signatures and the banding technique, suitable for Jaccard similarity.
        • __init__(self, num_bands, rows_per_band): Initializes parameters, including the number of bands and the number of hash rows per band.
        • index(self, signatures): Splits signatures into bands and stores them in hash buckets.
        • query(self, signature): Finds potentially similar documents by matching bands in buckets.
        • get_candidate_pairs(self): Returns all candidate document pairs found in the buckets.
    • Class SimHashLSHIndex
        • Functionality: Builds an LSH index using SimHash signatures and Hamming distance for approximate matching, suitable for text similarity.
        • __init__(self, radius): Initializes the allowable Hamming distance error radius (number of differing bits).
        • index(self, signatures): Maps each signature to multiple Hamming-neighbor buckets (via bit flipping).
        • query(self, signature): Enumerates the neighboring buckets of a signature to retrieve potentially similar documents.
        • get_candidate_pairs(self): Combines results from all buckets to output candidate document pairs.
    • Class BitSamplingLSHIndex
        • Functionality: Builds an LSH index using BitSampling signatures and multiple sampled hash tables, suitable for Hamming similarity.
        • __init__(self, num_hash_tables, bits_per_table): Initializes the number of hash tables and the number of sampled bits per table.
        • index(self, signatures): Performs multiple bit sampling on each document signature and stores the results in corresponding buckets.
        • query(self, signature): Hashes a given signature according to the sampling rules and retrieves documents from matching buckets.
        • get_candidate_pairs(self): Aggregates the contents of all hash tables to generate candidate document pairs.
    • Class HybridLSHIndex
        • Functionality: Builds an LSH index using hybrid MinHash and SimHash signatures, supporting various merging strategies.
        • __init__(self, minhash_params, simhash_params, merge_strategy, weights): Initializes parameters for the hybrid index.
        • index(self, signatures): Indexes document data containing both types of signatures.
        • query(self, signature): Queries potentially similar documents.
        • get_candidate_pairs(self): Retrieves candidate document pairs based on the merging strategy.
    • Usage: Called in main.py after obtaining all document signatures to build the LSH index and retrieve candidate pairs.
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
    LSH index based on MinHash signatures, suitable for Jaccard similarity.
    """

    def __init__(self, num_bands: int, rows_per_band: int):
        """
        Initializes the MinHash LSH index.
        Parameters:
            num_bands (int): Number of bands.
            rows_per_band (int): Number of hash rows per band.
        """
        self.num_bands = num_bands
        self.rows_per_band = rows_per_band
        self.buckets = [defaultdict(list) for _ in range(num_bands)]
        # Precompute the slice boundaries for each band, formatted as (start, end)
        self.band_indices = [
            (band_idx * self.rows_per_band, (band_idx + 1) * self.rows_per_band)
            for band_idx in range(num_bands)
        ]

    def _hash_band(self, band: Tuple[int]) -> int:
        """
        Hashes a band.
        Parameters:
            band (Tuple[int]): The content of the band.
        Returns:
            int: The hash value.
        """
        return hash(band)

    def index(self, signatures: List[List[int]]):
        """
        Splits signatures into bands and stores them in hash buckets.
        Parameters:
            signatures (List[List[int]]): List of MinHash signatures.
        """
        for doc_id, signature in tqdm(enumerate(signatures), desc="Indexing signatures", total=len(signatures)):
            # If possible, convert the signature to a tuple in advance to avoid converting for each band
            for band_idx, (start, end) in enumerate(self.band_indices):
                band = tuple(signature[start:end])
                bucket_key = self._hash_band(band)
                self.buckets[band_idx][bucket_key].append(doc_id)

    def query(self, signature: List[int]) -> Set[int]:
        """
        Queries potentially similar documents.
        Parameters:
            signature (List[int]): The MinHash signature to query.
        Returns:
            Set[int]: A set of potentially similar document IDs.
        """
        candidates = set()
        for band_idx, (start, end) in enumerate(self.band_indices):
            band = tuple(signature[start:end])
            bucket_key = self._hash_band(band)
            candidates.update(self.buckets[band_idx].get(bucket_key, []))
        return candidates

    def get_candidate_pairs(self) -> Set[Tuple[int, int]]:
        """
        Returns all candidate document pairs.
        Returns:
            Set[Tuple[int, int]]: A set of candidate document pairs.
        """
        candidate_pairs = set()
        for band in tqdm(self.buckets, desc="Processing bands", total=len(self.buckets)):
            for bucket in band.values():
                if len(bucket) > 1:
                    # Use set comprehension to generate combinations and update the candidate pair set
                    candidate_pairs.update(
                        {tuple(sorted(pair)) for pair in itertools.combinations(bucket, 2)})
        return candidate_pairs
