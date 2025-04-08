'''
	•	功能：包含常用的辅助函数。
	•	示例函数：
	•	def jaccard_similarity(set1, set2): 计算Jaccard相似度。
	•	def compute_runtime(func, *args, **kwargs): 用于测量某个函数执行时间。
	•	调用：各模块如指纹对比时或评估时调用，增强代码可复用性。

'''
import time
from typing import Set, Callable, Any


def jaccard_similarity(set1: Set[Any], set2: Set[Any]) -> float:
    """
    计算两个集合的 Jaccard 相似度。
    
    参数:
        set1 (Set[Any]): 第一个集合。
        set2 (Set[Any]): 第二个集合。
    
    返回:
        float: Jaccard 相似度，范围在 [0, 1]。
    """
    if not set1 and not set2:
        return 1.0  # 如果两个集合都为空，定义相似度为 1
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union


def compute_runtime(func: Callable, *args, **kwargs) -> float:
    """
    测量某个函数的执行时间。
    
    参数:
        func (Callable): 要测量的函数。
        *args: 传递给函数的参数。
        **kwargs: 传递给函数的关键字参数。
    
    返回:
        float: 函数执行时间（秒）。
    """
    start_time = time.time()
    func(*args, **kwargs)
    end_time = time.time()
    return end_time - start_time


# 示例用法
if __name__ == "__main__":
    # 示例 1: 计算 Jaccard 相似度
    set_a = {"apple", "banana", "cherry"}
    set_b = {"banana", "cherry", "date"}
    similarity = jaccard_similarity(set_a, set_b)
    print(f"Jaccard 相似度: {similarity:.2f}")

    # 示例 2: 测量函数运行时间
    def example_function(n):
        return sum(range(n))

    runtime = compute_runtime(example_function, 1000000)
    print(f"函数运行时间: {runtime:.4f} 秒")