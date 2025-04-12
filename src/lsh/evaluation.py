'''
    • 功能：对LSH候选对以及整个去重系统进行评估，计算去重率以及运行时间等指标。
    • 主要类与函数：
    • Class Evaluator
    • __init__(self, candidate_pairs): 初始化候选对
    • compute_duplicate_rate(self): 计算近重复文档比率。
    • generate_report(self): 输出统计结果，并生成可视化图表（如调用matplotlib绘制图像）。
    • 调用：在主流程中，通过main.py调用该评估模块，对LSH得到的候选近重复对进行性能和质量评估。
'''
from typing import Set, Tuple
import time
import matplotlib.pyplot as plt


class Evaluator:
    """
    Evaluator 类，用于对 LSH 候选对以及整个去重系统进行评估。
    """

    def __init__(self, candidate_pairs: Set[Tuple[int, int]]):
        """
        初始化 Evaluator 实例。

        参数:
            candidate_pairs (Set[Tuple[int, int]]): LSH 生成的候选近重复文档对。
        """
        self.candidate_pairs = candidate_pairs

    def compute_duplicate_rate(self) -> float:
        """
        计算候选对中的近重复文档比率。

        返回:
            float: 候选对中的近重复文档比率。
        """
        if not self.candidate_pairs:
            return 0.0
        duplicate_count = len(self.candidate_pairs)
        return duplicate_count / len(self.candidate_pairs)

    def generate_report(self, runtime: float, output_dir: str = None) -> None:
        """
        输出统计结果，并生成可视化图表。

        参数:
            runtime (float): 运行时间。
        """
        print("评估报告:")
        print(f"运行时间: {runtime:.2f} 秒")
        print(f"候选对数量: {len(self.candidate_pairs)}")
        print(f"近重复文档比率: {self.compute_duplicate_rate():.2f}")

        # 可视化候选对数量
        labels = ["Duplicate Rate"]
        values = [self.compute_duplicate_rate()]

        plt.bar(labels, values, color=["blue"])
        plt.ylim(0, 1)
        plt.title("Duplicate Rate")
        plt.ylabel("Rate")
        plt.show()

        # 保存图表到文件
        if output_dir:
            plt.savefig(f"{output_dir}/duplicate_rate.png")
            print(f"图表已保存到: {output_dir}/duplicate_rate.png")
        else:
            print("未指定输出目录，图表未保存。")


# 示例用法
if __name__ == "__main__":
    # 示例候选对
    candidate_pairs = {(1, 2), (2, 3), (4, 5)}

    # 初始化评估器
    evaluator = Evaluator(candidate_pairs)

    # 计算运行时间
    start_time = time.time()
    runtime = time.time() - start_time

    # 生成报告
    evaluator.generate_report(runtime)