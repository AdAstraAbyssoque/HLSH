
import re


def preprocess_text(text: str) -> str:
    """
    对输入文本进行基础清洗。

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
    text = re.sub(r"[^\w\s]", "", text)
    # 转换为小写
    text = text.lower()
    return text
