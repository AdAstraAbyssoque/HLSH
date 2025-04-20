from joblib import Parallel, delayed
from tqdm import tqdm
import re
from typing import List
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

def preprocess_text(text: str) -> str:
    """
    Performs basic cleaning on the input text.

    Parameters:
        text (str): Raw text.

    Returns:
        str: Cleaned text.
    """
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()
    # Remove punctuation
    text = re.sub(r"[^\w\s]", "", text)
    # Convert to lowercase
    text = text.lower()
    return text


def parallel_preprocess_texts(texts: list[str], process_pool_size: int, parallel_enabled: bool) -> list[str]:
    """
    Preprocesses a list of texts in parallel or sequentially.

    Parameters:
        texts (list[str]): List of raw texts.
        process_pool_size (int): Number of processes for parallel processing.
        parallel_enabled (bool): Whether to enable parallel processing.

    Returns:
        list[str]: List of preprocessed texts.
    """
    if parallel_enabled:
        print(f"Parallel processing enabled, number of processes: {process_pool_size}")
        preprocessed_data = Parallel(n_jobs=process_pool_size, prefer="processes")(
            delayed(preprocess_text)(text) for text in tqdm(texts, desc="Parallel preprocessing")
        )
    else:
        preprocessed_data = [preprocess_text(text) for text in tqdm(texts, desc="Preprocessing")]
    return preprocessed_data

class Preprocessor:
    """
    Text preprocessing class.
    Provides text cleaning, tokenization, and n-gram generation functionalities.
    """

    def __init__(self, config: dict):
        """
        Initializes the preprocessor.

        Parameters:
            config (dict): Configuration dictionary containing preprocessing parameters.
                - remove_punctuation (bool): Whether to remove punctuation, default is True.
                - lowercase (bool): Whether to convert text to lowercase, default is True.
                - stopwords (set): Completely overrides the default stopword list if provided.
                - extra_stopwords (set): Additional stopwords to add to the default list.
        """
        self.remove_punctuation = config.get("remove_punctuation", True)
        self.lowercase = config.get("lowercase", True)
        
        # Handle stopword configuration
        if "stopwords" in config:
            # If stopwords are provided, override the default stopword list
            self.stopwords = set(config["stopwords"])
        else:
            # Otherwise, use the default English stopwords
            self.stopwords = set(ENGLISH_STOP_WORDS)
            # Add extra stopwords if provided
            if "extra_stopwords" in config:
                self.stopwords.update(config["extra_stopwords"])

    def clean_text(self, text: str) -> str:
        """
        Performs basic cleaning and stopword filtering on the input text.

        Parameters:
            text (str): Raw text.

        Returns:
            str: Cleaned text.
        """
        # Remove HTML tags
        text = re.sub(r"<[^>]+>", "", text)
        # Remove extra spaces
        text = re.sub(r"\s+", " ", text).strip()
        # Remove punctuation
        if self.remove_punctuation:
            text = re.sub(r"[^\w\s]", "", text)
        # Convert to lowercase
        if self.lowercase:
            text = text.lower()
        # Filter stopwords
        if self.stopwords:
            tokens = text.split()
            filtered_tokens = [token for token in tokens if token not in self.stopwords]
            text = " ".join(filtered_tokens)
        # Handle very short sentences
        if len(text.split()) < 3:
            text = text + " This is A Filling text"
        return text

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenizes the text.

        Parameters:
            text (str): Cleaned text.

        Returns:
            List[str]: List of tokens.
        """
        tokens = text.split()
        # Remove stopwords
        if self.stopwords:
            tokens = [token for token in tokens if token not in self.stopwords]
            return tokens
        return tokens

    def generate_ngrams(self, tokens: List[str], n: int) -> List[str]:
        """
        Generates n-gram sequences from tokens.

        Parameters:
            tokens (List[str]): List of tokens.
            n (int): The value of n for n-grams.

        Returns:
            List[str]: List of n-grams.
        """
        ngrams = [" ".join(tokens[i:i + n])
                  for i in range(len(tokens) - n + 1)]
        return ngrams


def preprocess_dataset(data_list: List[str], config: dict) -> List[dict]:
    """
    Batch preprocesses a list of data.

    Parameters:
        data_list (List[str]): List of raw text data.
        config (dict): Configuration dictionary containing preprocessing parameters.

    Returns:
        List[dict]: List of preprocessed documents, each containing cleaned text, tokens, and n-grams.
    """
    preprocessor = Preprocessor(config)
    processed_data = []

    for text in data_list:
        cleaned_text = preprocessor.clean_text(text)
        tokens = preprocessor.tokenize(cleaned_text)
        ngrams = preprocessor.generate_ngrams(
            tokens, config.get("n", 2))  # Default to generating bi-grams
        processed_data.append({
            "cleaned_text": cleaned_text,
            "tokens": tokens,
            "ngrams": ngrams
        })

    return processed_data


# Example usage
if __name__ == "__main__":
    # Example configuration
    config = {
        "remove_punctuation": True,
        "lowercase": True,
        "stopwords": {"the", "is", "and", "in", "to"},
        "n": 3  # Generate tri-grams
    }

    # Example data
    raw_data = [
        "This is an example sentence.",
        "Another example, with punctuation!",
        "<p>HTML tags should be removed.</p>"
    ]

    # Batch preprocessing
    processed = preprocess_dataset(raw_data, config)
    for item in processed:
        print(type(item["ngrams"]))
        print(type(item["tokens"]))
        print(type(item["cleaned_text"]))