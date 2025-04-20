'''
        • Functionality: Converts preprocessed text (tokens, ngrams -> list[str], cleaned_text -> str) into feature formats suitable for fingerprint computation (minhash, simhash, bitsampling), such as n-gram sets, token sets, or vector representations.
        • Main classes and functions:
        • Class FeatureExtractor
        • __init__(self, method="ngram", **kwargs): Determines which feature extraction strategy to use based on the method and parameters.
        • extract_features(self, text): Extracts features from a single text (returns n-gram sets or token sets).
        • Usage: Called in main.py after preprocessing to prepare input for subsequent fingerprint computation.
'''
from typing import List, Set, Union
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from joblib import Parallel, delayed
from tqdm import tqdm


class FeatureExtractor:
    """
    Feature extractor class for converting text into feature formats suitable for fingerprint computation.
    Supports n-gram sets, token sets, or vectorized representations (using a Vectorizer).
    """

    def __init__(self, method: str = "ngram", n: int = 3, **kwargs):
        """
        Initializes the feature extractor.

        Parameters:
            method (str): Feature extraction method, supports "ngram", "token", "frequency", and "vectorize".
            n (int): The value of n for n-grams (only effective when method="ngram").
            **kwargs: Parameters for the vectorization method, can be passed to the vectorizer.
        """
        self.method = method
        self.n = n
        self.vectorizer = kwargs.pop("vectorizer", None)
        if method == "vectorize" and self.vectorizer is None:
            # Default to TfidfVectorizer, can be replaced with CountVectorizer if needed
            self.vectorizer = TfidfVectorizer(**kwargs)
            # The vectorizer needs to be fitted to a corpus; we delay fitting for later single-document conversion

    def extract_features(self, text: str) -> Set[str]:
        """
        Extracts a set of features from a single text. The output is standardized to a set[str] format,
        which can be directly used as input for BitSampling.

        Parameters:
            text (str): Input text.

        Returns:
            Set[str]: Extracted feature set.
        """
        if self.method == "ngram":
            return self._extract_ngrams(text)
        elif self.method == "token":
            return self._extract_tokens(text)
        elif self.method == "frequency":
            return self._extract_frequency(text)
        else:
            raise ValueError(
                f"Unsupported feature extraction method: {self.method}")

    def parallel_extract_features(self, texts: List[str], process_pool_size: int, parallel_enabled: bool) -> List[Set[str]]:
        """
        Extracts features from text in parallel or sequentially.

        Parameters:
            texts (List[str]): List of input texts.
            process_pool_size (int): Number of processes for parallel processing.
            parallel_enabled (bool): Whether to enable parallel processing.

        Returns:
            List[Set[str]]: List of extracted feature sets.
        """
        if self.method == "vectorize":
            # If the method is vectorize, directly use the vectorizer for transformation
            return self._extract_vectorized_features(texts)
        # Otherwise, use the extract_features method to extract features
        if parallel_enabled:
            print(
                f"Parallel feature extraction enabled, number of processes: {process_pool_size}")
            features = Parallel(n_jobs=process_pool_size, prefer="processes")(
                delayed(self.extract_features)(text) for text in tqdm(texts, desc="Parallel feature extraction")
            )
        else:
            features = [self.extract_features(text) for text in tqdm(
                texts, desc="Feature extraction")]
        return features

    def _extract_ngrams(self, text: str) -> Set[str]:
        """
        Extracts n-gram feature sets.

        Parameters:
            text (str): Input text.

        Returns:
            Set[str]: n-gram set.
        """
        tokens = text.split()  # Simple tokenization
        ngrams = set(
            [" ".join(tokens[i:i + self.n])
             for i in range(len(tokens) - self.n + 1)]
        )
        return ngrams

    def _extract_tokens(self, text: str) -> Set[str]:
        """
        Extracts token feature sets.

        Parameters:
            text (str): Input text.

        Returns:
            Set[str]: Token set.
        """
        return set(text.split())

    def _extract_frequency(self, text: str) -> dict:
        """
        Extracts the frequency of words from the given text.

        This method splits the input text into words and calculates the frequency
        of each word using a Counter.

        Args:
            text (str): The input text from which to extract word frequencies.

        Returns:
            dict: A dictionary of word frequencies.
        """
        return dict(Counter(text.split()))

    def _extract_vectorized_features(self, documents: List[str]) -> dict:
        """
        Uses a pre-fitted or currently initialized TfidfVectorizer to convert the input document list
        into a TF-IDF dictionary representation, with tokens as keys and their cumulative TF-IDF scores as values.

        Parameters:
            documents (List[str]): List of input documents.

        Returns:
            dict: A dictionary with tokens as keys and their cumulative TF-IDF scores as values.
        """
        try:
            # Attempt direct transformation (assuming the vectorizer is already fitted)
            print("Attempting direct TF-IDF matrix transformation")
            tfidf_matrix = self.vectorizer.transform(documents)
        except Exception:
            # If not fitted, fit first and then transform
            print("Vectorizer not fitted, attempting to fit TF-IDF matrix")
            tfidf_matrix = self.vectorizer.fit_transform(documents)

        print("TF-IDF matrix transformation successful, extracting features")
        feature_names = self.vectorizer.get_feature_names_out()
        # Use matrix operations to accumulate TF-IDF scores for each token
        # Sum scores for each column (token)
        token_scores = tfidf_matrix.sum(axis=0)
        # Convert results to a 1D array (if tfidf_matrix is sparse)
        token_scores = token_scores.A1
        token_dict = dict(zip(feature_names, token_scores))
        return token_dict


# Example usage
if __name__ == "__main__":
    corpus = [
        "this is a test",
        "this is another test",
        "this is a simple example for feature extraction",
        "this is another example for feature extraction",
        "this is a test for feature extraction"
    ]
    text = "this is a simple example for feature extraction"

    # Example 1: Using n-gram features
    extractor_ngram = FeatureExtractor(method="ngram", n=2)
    ngram_features = extractor_ngram.extract_features(text)
    print("n-gram features:", ngram_features)

    # Example 2: Using token features
    extractor_token = FeatureExtractor(method="token")
    token_features = extractor_token.extract_features(text)
    print("Token features:", token_features)

    # Example 3: Using vectorized features (TF-IDF non-zero feature set)
    extractor_vectorize = FeatureExtractor(
        method="vectorize", vectorizer=TfidfVectorizer())
    vectorized_features = extractor_vectorize.extract_features(text)
    print("Vectorized features:", vectorized_features)
