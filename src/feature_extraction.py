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
    支持 n-gram 集合、token 集合或 vectorize（利用 Vectorizer 提取特征集合）。
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
        return set(text.split())  # 简单分词后去重

    def _extract_vectorized_features(self, text: str) -> Set[str]:
        """
        利用 vectorizer 将文本转换为特征集合。
        基于 vectorizer 的非零（或重要）特征构造集合，
        注意：需要先对语料进行拟合，这里简单实现为对单个文档进行转换。

        参数:
            text (str): 输入文本。

        返回:
            Set[str]: 特征集合，来自 vectorizer 的特征名称。
        """
        # 如果 vectorizer 未拟合，则先拟合单个文档
        # 实际使用中建议先调用 fit 方法拟合语料库
        try:
            # 尝试直接转换
            sparse = self.vectorizer.transform([text])
        except Exception:
            # 若未拟合，则先拟合后转换
            self.vectorizer.fit([text])
            sparse = self.vectorizer.transform([text])
        feature_names = self.vectorizer.get_feature_names_out()
        indices = sparse.nonzero()[1]
        return {feature_names[i] for i in indices}

    def vectorize(self, documents: List[str]):
        """
        将文档批量转换为稀疏/密集矩阵（直接调用 vectorizer）。

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
    vectorized_features = extractor_vectorize.extract_features(text)
    print("vectorize 特征:", vectorized_features)