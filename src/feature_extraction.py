'''
        •	功能：将预处理后的文本(tokens,ngrams->list[str], cleaned_text->str)转换为适合指纹计算(minhash, simhash, bitsampling)的特征格式，如n-gram集合、token集合或向量表示。
        •	主要类与函数：
        •	Class FeatureExtractor
        •	__init__(self, method="ngram", **kwargs): 根据方法和参数决定采用哪种特征提取策略。
        •	extract_features(self, text): 从单个文本中提取特征（返回n-gram集合或token集合）。
        •	vectorize(self, documents): 可选，用于批量将文档转换为稀疏/密集向量，支持如CountVectorizer或TfidfVectorizer的调用。
        •	调用：main.py在预处理后调用此模块为后续指纹计算准备输入。
'''
from typing import List, Set, Union
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


class FeatureExtractor:
    """
    特征提取器类，用于将文本转换为适合指纹计算的特征格式。
    支持 n-gram 集合、token 集合或向量表示。
    """

    def __init__(self, method: str = "ngram", n: int = 3, **kwargs):
        """
            初始化特征提取器。

            参数:
                    method (str): 特征提取方法，支持 "ngram"、"token" 和 "vectorize"。
                    n (int): n-gram 的 n 值（仅在 method="ngram" 时有效）。
                    **kwargs: 向量化方法的参数，可传入 vectorizer 或 CountVectorizer/TfidfVectorizer 的参数。
            """
        self.method = method
        self.n = n
        self.vectorizer = kwargs.pop("vectorizer", None)
        if method == "vectorize" and self.vectorizer is None:
            self.vectorizer = CountVectorizer(**kwargs)

    def extract_features(self, text: str) -> Set[str]:
        """
        从单个文本中提取特征。

        参数:
            text (str): 输入文本。

        返回:
            Set[str]: 提取的特征（n-gram 集合或 token 集合）。
        """
        if self.method == "ngram":
            return self._extract_ngrams(text)
        elif self.method == "token":
            return self._extract_tokens(text)
        else:
            raise ValueError(f"不支持的特征提取方法: {self.method}")

    def _extract_ngrams(self, text: str) -> Set[str]:
        """
        提取 n-gram 集合。

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
        提取 token 集合。

        参数:
            text (str): 输入文本。

        返回:
            Set[str]: token 集合。
        """
        return set(text.split())  # 简单分词后去重

    def vectorize(self, documents: List[str]):
        """
        将文档批量转换为稀疏/密集向量。

        参数:
            documents (List[str]): 文档列表。

        返回:
            稀疏矩阵或密集矩阵（取决于 vectorizer 的配置）。
        """
        if self.method != "vectorize" or not self.vectorizer:
            raise ValueError(
                "当前配置不支持向量化操作，请确保 method='vectorize' 且已初始化 vectorizer。")
        return self.vectorizer.fit_transform(documents)


# 示例用法
if __name__ == "__main__":
    # 示例 1: 提取 n-gram 特征
    extractor_ngram = FeatureExtractor(method="ngram", n=2)
    text = "this is a simple example"
    ngram_features = extractor_ngram.extract_features(text)
    print("n-gram 特征:", ngram_features)

    # 示例 2: 提取 token 特征
    extractor_token = FeatureExtractor(method="token")
    token_features = extractor_token.extract_features(text)
    print("token 特征:", token_features)

    # 示例 3: 文档向量化
    documents = ["this is the first document",
                 "this document is the second document"]
    extractor_vectorize = FeatureExtractor(
        method="vectorize", vectorizer=TfidfVectorizer())
    vectorized_features = extractor_vectorize.vectorize(documents)
    print("向量化特征（稀疏矩阵）:\n", vectorized_features.toarray())
