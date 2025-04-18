'''
        •	功能：提供文本预处理功能。
        •	主要类与函数：
        •	Class Preprocessor
        •	__init__(self, config): 根据配置初始化预处理参数（例如是否需要去除标点、大小写标准化、停用词列表等）。
        •	clean_text(self, text): 对输入文本进行基础清洗（去除噪声符号、HTML标签等）。
        •	tokenize(self, text): 实现词或字符的分词。
        •	generate_ngrams(self, tokens, n): 基于tokens生成n-gram序列。
        •	def preprocess_dataset(data_list, config)
        •	循环调用Preprocessor的方法，返回预处理后的文档列表或文档对应的特征集合。
        •	调用：main.py在加载数据后调用该模块进行文本的基础预处理。

'''

import re
from typing import List
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

class Preprocessor:
    """
    文本预处理类。
    提供文本清洗、分词和 n-gram 生成功能。
    """

    def __init__(self, config: dict):
        """
        初始化预处理器。

        参数:
            config (dict): 配置字典，包含预处理参数。
                - remove_punctuation (bool): 是否去除标点符号，默认为True。
                - lowercase (bool): 是否将文本转换为小写，默认为True。
                - stopwords (set): 完全覆盖默认停用词列表，如果提供。
                - extra_stopwords (set): 在默认停用词基础上额外添加的停用词。
        """
        self.remove_punctuation = config.get("remove_punctuation", True)
        self.lowercase = config.get("lowercase", True)
        
        # 处理停用词配置
        if "stopwords" in config:
            # 如果配置了stopwords，完全覆盖默认停用词
            self.stopwords = set(config["stopwords"])
        else:
            # 否则使用默认英语停用词
            self.stopwords = set(ENGLISH_STOP_WORDS)
            # 添加额外停用词
            if "extra_stopwords" in config:
                self.stopwords.update(config["extra_stopwords"])

    def clean_text(self, text: str) -> str:
        """
        对输入文本进行基础清洗和停用词过滤。

        参数:
            text (str): 原始文本。

        返回:
            str: 清洗后的文本。
        """
        # 去除HTML标签
        text = re.sub(r"<[^>]+>", "", text)
        # 去除多余的空格
        text = re.sub(r"\s+", " ", text).strip()
        # 去除标点符号
        if self.remove_punctuation:
            text = re.sub(r"[^\w\s]", "", text)
        # 转换为小写
        if self.lowercase:
            text = text.lower()
        # 过滤停用词
        if self.stopwords:
            tokens = text.split()
            filtered_tokens = [token for token in tokens if token not in self.stopwords]
            text = " ".join(filtered_tokens)
        # 处理过短的句子
        if len(text.split()) < 3:
            text = text + " This is A Filling text"
        return text

    def tokenize(self, text: str) -> List[str]:
        """
        对文本进行分词。

        参数:
            text (str): 清洗后的文本。

        返回:
            List[str]: 分词后的词列表。
        """
        tokens = text.split()
        # 去除停用词
        if self.stopwords:
            tokens = [token for token in tokens if token not in self.stopwords]
            return tokens
        return tokens

    def generate_ngrams(self, tokens: List[str], n: int) -> List[str]:
        """
        基于 tokens 生成 n-gram 序列。

        参数:
            tokens (List[str]): 分词后的词列表。
            n (int): n-gram 的 n 值。

        返回:
            List[str]: n-gram 序列。
        """
        ngrams = [" ".join(tokens[i:i + n])
                  for i in range(len(tokens) - n + 1)]
        return ngrams


def preprocess_dataset(data_list: List[str], config: dict) -> List[dict]:
    """
    对数据列表进行批量预处理。

    参数:
        data_list (List[str]): 原始文本数据列表。
        config (dict): 配置字典，包含预处理参数。

    返回:
        List[dict]: 预处理后的文档列表，每个文档包含清洗文本、分词和 n-gram。
    """
    preprocessor = Preprocessor(config)
    processed_data = []

    for text in data_list:
        cleaned_text = preprocessor.clean_text(text)
        tokens = preprocessor.tokenize(cleaned_text)
        ngrams = preprocessor.generate_ngrams(
            tokens, config.get("n", 2))  # 默认生成 bi-grams
        processed_data.append({
            "cleaned_text": cleaned_text,
            "tokens": tokens,
            "ngrams": ngrams
        })

    return processed_data


# 示例用法
if __name__ == "__main__":
    # 示例配置
    config = {
        "remove_punctuation": True,
        "lowercase": True,
        "stopwords": {"the", "is", "and", "in", "to"},
        "n": 3  # 生成 tri-grams
    }

    # 示例数据
    raw_data = [
        "This is an example sentence.",
        "Another example, with punctuation!",
        "<p>HTML tags should be removed.</p>"
    ]

    # 批量预处理
    processed = preprocess_dataset(raw_data, config)
    for item in processed:
        # print(item)
        print(type(item["ngrams"]))
        print(type(item["tokens"]))
        print(type(item["cleaned_text"]))
