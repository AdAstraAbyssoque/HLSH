'''
    • Functionality: Implements the MinHash algorithm, generates MinHash signatures, and provides similarity evaluation functions based on signatures.
    • Main classes and functions:
    • Class MinHash
    • __init__(self, num_hashes, seed=None): Initializes the number of hash functions and random seed.
    • compute_signature(self, feature_set): Takes input like n-grams or token sets and outputs fixed-length MinHash signatures.
    • compare_signatures(self, sig1, sig2): Computes approximate Jaccard similarity between two signatures.
    • Usage: Called in main.py to generate signatures for preprocessed features. Typically used with the banding technique in LSH.
'''
import random
import numpy as np
from typing import List, Set
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm


class MinHash:
    """
    MinHash class for generating MinHash signatures and computing similarity between signatures.
    """

    def __init__(self, num_hashes: int, seed: int = None):
        """
        Initializes a MinHash instance.

        Parameters:
            num_hashes (int): Number of hash functions.
            seed (int): Random seed for generating hash functions (default is None).
        """
        self.num_hashes = num_hashes
        self.seed = seed
        self.params = self._generate_hash_params()  # Stores (a, b, p) tuples

    def _generate_hash_params(self) -> List[tuple]:
        """
        Generates a set of hash function parameters, each as (a, b, p).

        Returns:
            List[tuple]: List of hash parameters.
        """
        if self.seed is not None:
            random.seed(self.seed)
        params = []
        p = 2**33 - 355  # A large prime number
        for _ in range(self.num_hashes):
            a = random.randint(1, 2**32 - 1)
            b = random.randint(0, 2**32 - 1)
            params.append((a, b, p))
        return params

    def compute_signature(self, feature_set: Set[str]) -> List[int]:
        """
        Computes the MinHash signature.

        Parameters:
            feature_set (Set[str]): Input feature set (e.g., n-gram set).

        Returns:
            List[int]: Fixed-length MinHash signature.
        """
        # Precompute the base hash values for each feature to avoid repeated calls to hash(feature)
        hash_values = np.array([hash(feature) for feature in feature_set])
        signature = []

        # Use NumPy vectorization to batch compute hash values for each (a, b, p) parameter
        for a, b, p in self.params:
            # Compute hash values for all features: (a * hash(feature) + b) mod p
            transformed = (a * hash_values + b) % p
            # Take the minimum value as the output of the current hash function
            signature.append(int(transformed.min()))
        return signature

    def parallel_compute_signature(self, feature_sets: List[Set[str]], parallel_enable: bool, process_pool_size: int) -> List[List[int]]:
        """
        Computes MinHash signatures for multiple feature sets in parallel.

        Parameters:
            feature_sets (List[Set[str]]): List of input feature sets.
            process_pool_size (int): Number of processes for parallel computation.
            parallel_enable (bool): Whether to enable parallel processing.

        Returns:
            List[List[int]]: List of MinHash signatures for each feature set.
        """
        signatures = []
        if not parallel_enable:
            signatures = [self.compute_signature(feature_set) for feature_set in tqdm(
                feature_sets, desc="Generating MinHash signatures")]
        else:
            # Use ProcessPoolExecutor for parallel computation
            with ProcessPoolExecutor(max_workers=process_pool_size) as executor:
                signatures = list(tqdm(
                    executor.map(self.compute_signature, feature_sets),
                    total=len(feature_sets),
                    desc="Generating MinHash signatures in parallel"
                ))
        return signatures


def compare_signatures(sig1: List[int], sig2: List[int]) -> float:
    """
    Compares the similarity between two MinHash signatures.

    Parameters:
        sig1 (List[int]): First MinHash signature.
        sig2 (List[int]): Second MinHash signature.

    Returns:
        float: Approximate Jaccard similarity.
    """
    if len(sig1) != len(sig2):
        raise ValueError("Signature lengths do not match, comparison is not possible.")
    # Use a generator expression to calculate the match rate
    return sum(1 for i in range(len(sig1)) if sig1[i] == sig2[i]) / len(sig1)


# Example usage
if __name__ == "__main__":
    # Example feature sets
    feature_set1 = {"thi", "his", "is ", "s i", " is",
                    "is ", "s a", " a ", "a t", " te", "tes", "est"}
    feature_set2 = {"thi", "his", "is ", "s i", " is", "is ", "s a",
                    " a ", "a t", " te", "tes", "est", "ano", "not", "oth", "the", "her"}

    # Initialize MinHash
    minhash = MinHash(num_hashes=100, seed=42)

    # Compute signatures
    signature1 = minhash.compute_signature(feature_set1)
    signature2 = minhash.compute_signature(feature_set2)

    # Compare signature similarity
    similarity = compare_signatures(signature1, signature2)
    print("MinHash Signature 1:", signature1)
    print("MinHash Signature 2:", signature2)
    print("MinHash Signature Similarity:", similarity)