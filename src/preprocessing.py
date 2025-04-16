from joblib import Parallel, delayed
from tqdm import tqdm
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


def parallel_preprocess_texts(texts: list[str], process_pool_size: int, parallel_enabled: bool) -> list[str]:
    """
    并行或串行预处理文本列表。

    参数:
        texts (list[str]): 原始文本列表。
        process_pool_size (int): 并行处理的进程数。
        parallel_enabled (bool): 是否启用并行处理。

    返回:
        list[str]: 预处理后的文本列表。
    """
    if parallel_enabled:
        print(f"并行已启用，进程数：{process_pool_size}")
        preprocessed_data = Parallel(n_jobs=process_pool_size, prefer="processes")(
            delayed(preprocess_text)(text) for text in tqdm(texts, desc="并行预处理")
        )
    else:
        preprocessed_data = [preprocess_text(text) for text in tqdm(texts, desc="预处理")]
    return preprocessed_data