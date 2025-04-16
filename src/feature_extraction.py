'''
        •	功能：将预处理后的文本(tokens,ngrams->list[str], cleaned_text->str)转换为适合指纹计算(minhash, simhash, bitsampling)的特征格式，如n-gram集合、token集合或向量表示。
        •	主要类与函数：
        •	Class FeatureExtractor
        •	__init__(self, method="ngram", **kwargs): 根据方法和参数决定采用哪种特征提取策略。
        •	extract_features(self, text): 从单个文本中提取特征（返回n-gram集合或token集合）。
        •	调用：main.py在预处理后调用此模块为后续指纹计算准备输入。
'''
from typing import List, Set, Union
from sklearn.feature_extraction.text import  TfidfVectorizer
from collections import Counter
class FeatureExtractor:
    """
    特征提取器类，用于将文本转换为适合指纹计算的特征格式。
    支持 n-gram 集合、token 集合或 vectorize（利用 Vectorizer 提取特征集合）。
    """

    def __init__(self, method: str = "ngram", n: int = 3, **kwargs):
        """
            初始化特征提取器。

            参数:
                method (str): 特征提取方法，支持 "ngram"、"token" 和 "vectorize"。
                n (int): n-gram 的 n 值（仅在 method="ngram" 时有效）。
                **kwargs: 向量化方法的参数，可传入 vectorizer  的参数。
            """
        self.method = method
        self.n = n
        self.vectorizer = kwargs.pop("vectorizer", None)
        if method == "vectorize" and self.vectorizer is None:
            # 默认使用 TfidfVectorizer，也可以根据需要替换 CountVectorizer
            self.vectorizer = TfidfVectorizer(**kwargs)
            # 需要先拟合语料库，此处为了后续单个文档转换，我们延迟拟合

    def extract_features(self, text: str) -> Set[str]:
        """
        从单个文本中提取特征集合，这里统一输出 set[str] 格式，
        可直接用于 BitSampling 的输入。

        参数:
            text (str): 输入文本。

        返回:
            Set[str]: 提取的特征集合。
        """
        if self.method == "ngram":
            return self._extract_ngrams(text)
        elif self.method == "token":
            return self._extract_tokens(text)
        elif self.method == "vectorize":
            return self._extract_vectorized_features(text)
        else:
            raise ValueError(f"不支持的特征提取方法: {self.method}")

    def _extract_ngrams(self, text: str) -> Set[str]:
        """
        提取 n-gram 特征集合。

        参数:
            text (str): 输入文本。

        返回:
            Set[str]: n-gram 集合。
        """
        tokens = text.split()  # 简单分词
        ngrams = set(
            [" ".join(tokens[i:i + self.n])
             for i in range(len(tokens) - self.n + 1)]
        )
        return ngrams

    def _extract_tokens(self, text: str) -> Set[str]:
        """
        提取 token 特征集合。

        参数:
            text (str): 输入文本。

        返回:
            Set[str]: token 集合。
        """
        return set(text.split())
        
    def _extract_frequency(self, text: str) -> Set[str]:
        """
        Extracts the frequency of words from the given text.

        This method splits the input text into words and calculates the frequency
        of each word using a Counter.

        Args:
            text (str): The input text from which to extract word frequencies.

        Returns:
            Set[str]: A set of unique words from the text with their frequencies.
        """
        return Counter(text.split())
    
    def _extract_tfidf(self, corpus: list[str]):
        """
        计算文本语料库中每个词的逆文档频率（IDF）。

        参数:
            corpus (list[str]): 文本语料库，包含多个文档。

        返回:
            tfidf_matrix (sparse matrix): 稀疏矩阵，表示每个文档中每个词的 TF-IDF 值。
            words (list): 词汇表，包含语料库中所有唯一的词。
        """
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(corpus)
        words = vectorizer.get_feature_names_out()
        return tfidf_matrix, words

        

    def _extract_vectorized_features(self, documents: Union[str, List[str]]):
        """
        利用预先拟合或当前初始化的 TfidfVectorizer，
        将输入文档转换为 TF-IDF 矩阵表示。
        
        参数:
            documents (str 或 List[str]): 输入单个文档或文档列表。
        
        返回:
            tfidf_matrix (sparse matrix): 表示文档中每个词 TF-IDF 值的稀疏矩阵。
        """
        # 若传入单个文档，则构造单元素列表
        if isinstance(documents, str):
            documents = [documents]
        try:
            # 尝试直接转换（假设 vectorizer 已经拟合）
            tfidf_matrix = self.vectorizer.transform(documents)
        except Exception:
            # 若未拟合，则先拟合后转换
            tfidf_matrix = self.vectorizer.fit_transform(documents)
        words = self.vectorizer.get_feature_names_out()
        return tfidf_matrix, words




# 示例用法
if __name__ == "__main__":
    corpus=["this is a test", 
            "this is another test",
            "this is a simple example for feature extraction",
            "this is another example for feature extraction",
            "this is a test for feature extraction"
            ]
    text = "this is a simple example for feature extraction"
    
    # 示例 1: 使用 ngram 特征
    extractor_ngram = FeatureExtractor(method="ngram", n=2)
    ngram_features = extractor_ngram.extract_features(text)
    print("n-gram 特征:", ngram_features)

    # 示例 2: 使用 token 特征
    extractor_token = FeatureExtractor(method="token")
    token_features = extractor_token.extract_features(text)
    print("token 特征:", token_features)

    # 示例 3: 使用 vectorize 特征（转换为 TF-IDF 非零项特征集合）
    extractor_vectorize = FeatureExtractor(method="vectorize", vectorizer=TfidfVectorizer())
    vectorized_features,words = extractor_vectorize.extract_features(text)
    print("vectorized 特征:", vectorized_features)
    print("词汇表:", words)