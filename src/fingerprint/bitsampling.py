"""
    • Functionality: Implements the Bit Sampling technique.
    • Main classes and functions:
    • Class BitSampling
    • __init__(self, sample_size): Specifies the number of bits or proportion to sample.
    • vectorize(self, feature_set): Converts a feature set into a binary vector.
    • compute_signature(self, feature_set): Generates a binary signature by sampling the input features.
    • Usage: Called in main.py as needed for experiments and result comparisons.
"""
from typing import Set, List
import random


class BitSampling:
    """
    Implements the Bit Sampling technique for generating binary signatures.
    Supports optional TF-IDF vectorization (via a pre-fitted vectorizer).
    If no vectorizer is provided, a simple XOR aggregation is used.
    """

    def __init__(self, sample_size: int, hash_bits: int = 64, seed: int = None, vectorizer=None):
        """
        Initializes a BitSampling instance.

        Parameters:
            sample_size (int): Number of bits to sample.
            hash_bits (int): Number of bits in the input signature (default is 64 bits).
            seed (int): Random seed for generating sample bit indices (default is None).
            vectorizer: Optional TF-IDF vectorizer (e.g., TfidfVectorizer), must be pre-fitted on a corpus.
        """
        if sample_size > hash_bits:
            raise ValueError(
                "Sample size cannot exceed the number of input signature bits.")
        self.sample_size = sample_size
        self.hash_bits = hash_bits
        self.seed = seed
        self.vectorizer = vectorizer  # Optional TF-IDF vectorizer
        self.sample_indices = self._generate_sample_indices()

    def _generate_sample_indices(self) -> List[int]:
        """
        Generates the indices of the bits to sample.

        Returns:
            List[int]: List of sampled bit indices.
        """
        if self.seed is not None:
            random.seed(self.seed)
        return random.sample(range(self.hash_bits), self.sample_size)

    def vectorize(self, feature_set: Set[str]) -> int:
        """
        Optimized method to convert a feature set into a binary vector:
        If a TF-IDF vectorizer is provided, it directly uses vectorizer.vocabulary_ and idf_ to get weights,
        avoiding repeated calls to transform, thus improving efficiency.

        Parameters:
            feature_set (Set[str]): Input feature set.

        Returns:
            int: Binary vector representation.
        """
        if self.vectorizer is not None:
            weights = {}
            vocab = self.vectorizer.vocabulary_
            # idf_ is a numpy array, indices correspond to values in vocab
            idf = self.vectorizer.idf_
            for feature in feature_set:
                if feature in vocab:
                    weights[feature] = idf[vocab[feature]]
            return self.vectorize_weighted(feature_set, weights)
        else:
            # Simple XOR aggregation
            vector = 0
            for feature in feature_set:
                hashed = hash(feature) & ((1 << self.hash_bits) - 1)
                vector ^= hashed
            return vector

    def vectorize_weighted(self, feature_set: Set[str], feature_weights: dict) -> int:
        """
        Converts a feature set into a weighted binary vector.
        For each bit, calculates cumulative weights and generates binary representation
        based on the sign of the cumulative score. TF-IDF weights can be used here.

        Parameters:
            feature_set (Set[str]): Input feature set.
            feature_weights (dict): Dictionary of weights for each feature.

        Returns:
            int: Binary vector representation.
        """
        score = [0] * self.hash_bits
        for feature in feature_set:
            weight = feature_weights.get(feature, 1)
            # Hash the feature and update the score for each bit
            hashed = hash(feature)
            for i in range(self.hash_bits):
                bit = (hashed >> i) & 1
                score[i] += weight if bit else -weight
        vector = 0
        for i, val in enumerate(score):
            if val > 0:
                vector |= (1 << i)
        return vector

    def compute_signature(self, feature_set: Set[str], feature_weights: dict = None) -> str:
        """
        Generates a binary signature from the input feature set.
        If feature_weights are provided, uses weighted vectorization (e.g., TF-IDF),
        otherwise uses simple XOR aggregation.

        Parameters:
            feature_set (Set[str]): Input feature set.
            feature_weights (dict, optional): Dictionary of weights for each feature.

        Returns:
            str: Binary signature after sampling.
        """
        if feature_weights is not None:
            full_vector = self.vectorize_weighted(feature_set, feature_weights)
        else:
            full_vector = self.vectorize(feature_set)
        signature = 0
        for i, bit_index in enumerate(self.sample_indices):
            if full_vector & (1 << bit_index):
                signature |= (1 << i)
        return str(signature)

    def compare_signatures(self, sig1: int, sig2: int) -> float:
        """
        Compares the similarity between two sampled signatures.

        Parameters:
            sig1 (int): First sampled signature.
            sig2 (int): Second sampled signature.

        Returns:
            float: Similarity measure (between 0 and 1).
        """
        hamming_distance = bin(sig1 ^ sig2).count("1")
        return 1 - (hamming_distance / self.sample_size)


# Example usage
if __name__ == "__main__":
    from sklearn.feature_extraction.text import TfidfVectorizer

    # Pre-fit a TF-IDF vectorizer (example uses the current sample corpus)
    corpus = ["this is a test", "this is another test"]
    vectorizer = TfidfVectorizer()
    vectorizer.fit(corpus)

    # Example feature sets
    feature_set1 = {"this", "is", "a", "test"}
    feature_set2 = {"this", "is", "another", "test"}

    # Initialize BitSampling and pass the TF-IDF vectorizer
    bitsampling = BitSampling(
        sample_size=16, hash_bits=64, seed=42, vectorizer=vectorizer)

    # Compute sampled signatures
    signature1 = bitsampling.compute_signature(feature_set1)
    signature2 = bitsampling.compute_signature(feature_set2)

    # Print sampled signatures and their similarity
    print("Sampled Signature 1:", bin(signature1))
    print("Sampled Signature 2:", bin(signature2))
    print("Signature Similarity:",
          bitsampling.compare_signatures(signature1, signature2))
