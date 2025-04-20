'''
    • Functionality: Implements the SimHash algorithm.
    • Main classes and functions:
    • Class SimHash
    • __init__(self, hash_bits=64): Specifies the number of bits for the SimHash output (typically 64 or 128 bits).
    • _compute_frequency(self, frequency: dict[str, int]): Generates a SimHash signature based on feature frequencies.
    • _compute_tfidf(self, tokens: list[set[str]], idf): Generates a SimHash signature based on TF-IDF weights.
    • compute_signature(self, feature_set, idf=None): Takes a feature set as input and outputs a SimHash signature (binary representation).
    • hamming_distance(self, sig1, sig2): Computes the Hamming distance between two SimHash signatures for similarity comparison.
    • Usage: Can be called in main.py to generate fingerprints using the SimHash method and compare with other fingerprinting methods.
'''
import math
from collections import Counter
from typing import List, Set
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm


class SimHash:
    """
    SimHash class for generating SimHash signatures based on word frequency or TF-IDF weights,
    and for comparing signatures using Hamming distance. The output is a binary string.
    """

    def __init__(self, hash_bits: int = 64):
        """
        Initializes a SimHash instance and specifies the number of bits for the output signature.

        Parameters:
            hash_bits (int): Number of bits for the SimHash output (typically 64 or 128 bits).
        """
        self.hash_bits = hash_bits

    def _compute_frequency(self, frequency: dict) -> int:
        """
        Computes a SimHash signature based on word frequency.

        Parameters:
            frequency (dict): A dictionary where keys are words and values are their frequencies in the text.

        Returns:
            int: The computed SimHash signature (integer representation).
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
        Computes a SimHash signature based on TF-IDF weights.
        For each token, if it exists in the idf dictionary, its corresponding idf weight is used; otherwise, the default weight is 1.

        Parameters:
            feature_set (set): Input feature set (e.g., tokens or n-grams).
            idf (dict): A dictionary of idf values for each token, in the form {token: idf_value}.

        Returns:
            int: The computed SimHash signature (integer representation).
        """
        frequency = {token: idf.get(token, 1) for token in feature_set}
        return self._compute_frequency(frequency)

    def compute_signature(self, feature_set: set, idf: dict = None) -> str:
        """
        Generates a SimHash signature based on the input feature set. If idf weights are provided, it uses TF-IDF;
        otherwise, it uses word frequency (all token weights are set to 1).
        The output is a binary string with a width of hash_bits.

        Parameters:
            feature_set (set): Input feature set.
            idf (dict, optional): A dictionary of idf values for each token.

        Returns:
            str: SimHash signature (binary string).
        """
        if idf is not None:
            sig = self._compute_tfidf(feature_set, idf)
        else:
            frequency = {token: 1 for token in feature_set}
            sig = self._compute_frequency(frequency)
        return format(sig, '0{}b'.format(self.hash_bits))

    def parallel_compute_signature(self, feature_sets: List[Set[str]], idf: dict = None, parallel_enable: bool = False, process_pool_size: int = 4) -> List[str]:
        """
        Computes SimHash signatures for multiple feature sets in parallel. The output is a list of binary strings.

        Parameters:
            feature_sets (List[Set[str]]): List of input feature sets.
            idf (dict, optional): A dictionary of idf values for each token.
            parallel_enable (bool): Whether to enable parallel computation.
            process_pool_size (int): Size of the process pool.

        Returns:
            List[str]: SimHash signatures for each feature set (as binary strings).
        """
        signatures = []
        if parallel_enable:
            with ProcessPoolExecutor(max_workers=process_pool_size) as executor:
                futures = []
                for feature_set in tqdm(feature_sets, desc="Submitting tasks to process pool"):
                    if idf is not None:
                        futures.append(executor.submit(
                            self._compute_tfidf, feature_set, idf))
                    else:
                        futures.append(executor.submit(self._compute_frequency, {
                                       token: 1 for token in feature_set}))
                for future in tqdm(futures, desc="Computing SimHash signatures in parallel"):
                    int_sig = future.result()
                    signatures.append(
                        format(int_sig, '0{}b'.format(self.hash_bits)))
        else:
            for feature_set in tqdm(feature_sets, desc="Computing SimHash signatures"):
                signatures.append(self.compute_signature(feature_set, idf))
        return signatures


def hamming_distance(sig1: str, sig2: str) -> int:
    """
    Computes the Hamming distance between two SimHash signatures (binary strings),
    i.e., the number of differing bits.

    Parameters:
        sig1 (str): First signature (binary string).
        sig2 (str): Second signature (binary string).

    Returns:
        int: Hamming distance between the two signatures.
    """
    # Convert binary strings to integers
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


# Example usage
if __name__ == "__main__":
    text = "This is a test, this test is simple and effective."
    tokens = set(text.lower().replace(',', '').replace('.', '').split())

    simhash = SimHash(hash_bits=64)

    # Example 1: Based on word frequency
    binary_sig1 = simhash.compute_signature(tokens)
    print("SimHash Signature (based on frequency):", binary_sig1)

    # Example 2: Based on TF-IDF weights
    idf = {token: 1.5 for token in tokens}
    binary_sig2 = simhash.compute_signature(tokens, idf=idf)
    print("SimHash Signature (based on TF-IDF):", binary_sig2)

    distance = hamming_distance(binary_sig1, binary_sig2)
    print("Hamming Distance:", distance)
