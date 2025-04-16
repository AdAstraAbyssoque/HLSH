
from collections import Counter
import math


def cosine_similarity(s1: str, s2: str) -> float:
    """
    计算两个字符串的余弦相似度，使用词袋模型表示。

    参数：
      s1: 第一个字符串
      s2: 第二个字符串

    返回：
      余弦相似度，范围在 [0, 1] 之间。
    """
    # 按空格切分字符串，构造词频向量
    words1 = s1.split()
    words2 = s2.split()
    vec1 = Counter(words1)
    vec2 = Counter(words2)

    # 求并集集合
    all_words = set(vec1.keys()).union(vec2.keys())

    # 计算向量点积
    dot_product = sum(vec1[word] * vec2[word] for word in all_words)

    # 计算向量的模
    norm1 = math.sqrt(sum(v ** 2 for v in vec1.values()))
    norm2 = math.sqrt(sum(v ** 2 for v in vec2.values()))

    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)


def jaccard_similarity(s1: str, s2: str) -> float:
    """
    计算两个字符串的 Jaccard 相似度，基于词集合进行计算。

    参数：
      s1: 第一个字符串
      s2: 第二个字符串

    返回：
      Jaccard 相似度，范围在 [0, 1] 之间。
    """
    # 按空格分词，构造集合
    set1 = set(s1.split())
    set2 = set(s2.split())

    # 计算交集与并集大小
    intersection = set1.intersection(set2)
    union = set1.union(set2)

    if not union:
        return 0.0
    return len(intersection) / len(union)


def euclidean_distance(s1: str, s2: str) -> float:
  """
  计算两个字符串的欧氏相似度，基于词袋模型表示。
  注意：较大的相似度值表示字符串之间更相似。

  参数：
    s1: 第一个字符串
    s2: 第二个字符串

  返回：
    欧氏相似度，范围在 [0, 1] 之间。
  """
  # 按空格分词，构造词频向量
  words1 = s1.split()
  words2 = s2.split()
  vec1 = Counter(words1)
  vec2 = Counter(words2)

  # 取两个向量的并集构建完整特征集合
  all_words = set(vec1.keys()).union(vec2.keys())

  # 计算欧氏距离的平方和（随后开平方）
  distance_sq = sum((vec1[word] - vec2[word]) ** 2 for word in all_words)
  distance = math.sqrt(distance_sq)

  # 转换为相似度，使用公式 1 / (1 + 距离)
  return 1 / (1 + distance)
