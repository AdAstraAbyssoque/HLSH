'''
• 功能：
	•	对整个 LSH 系统进行评估，主要包括两个方面：
	•	运行时性能对比：统计并分析各模块或阶段的运行时间，评估系统总体运行效率；
	•	近重复文档率计算：分析候选对结果，计算系统检测到的近重复（near-duplicate）文档比率。
	•	同时提供可视化支持，通过 matplotlib 绘制运行时对比图、近重复率分布图等，方便直观查看评估结果。
• 主要类与函数：
	•	Class LSHEvaluator
	•	init(self, candidate_pairs, runtime_log,data)
	    •	描述：初始化评估器，传入 LSH 生成的候选对、运行时日志数据和原始数据集。
	•	compute_runtime_comparison(self)
	    •	描述：统计并比较各阶段或模块的运行时数据，评估整体系统的计算效率，并生成相应的时间消耗报告。
	•	compute_near_duplicate_rate(self)
	    •	描述：使用相似度度量计算整个系统的近重复文档比率，分析候选对中近重复项的比例，评估系统检测质量。
	•	generate_visualizations(self)
	    •	描述：利用 matplotlib 生成可视化图表，包括但不限于：
	    •	运行时对比图（例如柱状图、折线图），展示各模块的运行时间；
	    •	近重复率分布图（例如直方图或饼图），直观展示近重复检测结果。
	•	generate_report(self)
    	•	描述：整合各项评估指标与图表，输出详细的评估报告。报告中应包含候选对总数、各模块运行时数据、近重复文档比率以及生成的可视化图表文件或图像预览。
• 调用：
	•	在主流程（例如 main.py 模块）中，实例化 LSHEvaluator 类，并传入 LSH 生成的候选对数据、运行时统计数据和其他辅助信息。
	•	调用 compute_runtime_comparison() 和 compute_near_duplicate_rate() 分别进行系统运行时和候选对质量的计算；
	•	调用 generate_visualizations() 绘制并保存图表；
	•	最后，通过 generate_report() 整合所有指标和图表，输出综合评估报告，帮助开发者或使用者直观了解 LSH 系统的性能和质量。
'''

import time
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from tqdm import tqdm



class Evaluator:
    """
    LSHEvaluator 类，用于对 LSH 系统进行评估，包括运行时性能对比和近重复文档率计算。
    """

    def __init__(self, candidate_pairs: List[Tuple[int, int]], runtime_log: Dict[str, float], data: List[str]):
        """
        初始化评估器。

        参数:
            candidate_pairs (List[Tuple[int, int]]): LSH 生成的候选对。
            runtime_log (Dict[str, float]): 各模块或阶段的运行时间。
                所含键值对格式为 {模块名: 运行时间}。
            data (List[str]): 原始数据集。

        """
        self.candidate_pairs = candidate_pairs
        self.runtime_log = runtime_log
        self.data = data

    def compute_runtime_comparison(self) -> Dict[str, float]:
        """
        统计并比较各阶段或模块的运行时数据。

        返回:
            Dict[str, float]: 各模块的运行时间。
        """
        print("运行时性能对比:")
        for module, runtime in self.runtime_log.items():
            print(f"{module}: {runtime:.2f} 秒")
        return self.runtime_log

    def compute_near_duplicate_rate(self, similarity_func) -> float:
        """
        使用相似度度量计算近重复文档比率。

        参数:
            similarity_func (Callable): 用于计算相似度的函数。

        返回:
            float: 近重复文档比率。
        """
        total_near_duplicates = 0
        for doc1_idx, doc2_idx in tqdm(self.candidate_pairs, desc="Calculating near-duplicate rate"):
            similarity = similarity_func(
                self.data[doc1_idx], self.data[doc2_idx])
            total_near_duplicates += similarity

        near_duplicate_rate = total_near_duplicates / \
            len(self.candidate_pairs) if self.candidate_pairs else 0.0
        return near_duplicate_rate

    def generate_visualizations(self, output_dir: str = None) -> None:
        """
        Generate visualizations using matplotlib.

        Parameters:
            output_dir (str): Directory to save the generated charts (optional).
        """
        # Runtime comparison chart
        modules = list(self.runtime_log.keys())
        runtimes = list(self.runtime_log.values())

        plt.figure(figsize=(10, 5))
        plt.bar(modules, runtimes, color='skyblue')
        plt.xlabel("Modules")
        plt.ylabel("Runtime (seconds)")
        plt.title("Runtime Performance Comparison")
        plt.xticks(rotation=45)
        plt.tight_layout()
        if output_dir:
            plt.savefig(f"{output_dir}/runtime_comparison.png")
            print(f"Runtime performance comparison chart saved to: {output_dir}/runtime_comparison.png")
        else:
            plt.show()

    def generate_report(self, similarity_func,  output_dir: str = None) -> None:
        """
        整合各项评估指标与图表，输出详细的评估报告。

        参数:
            similarity_func (Callable): 用于计算相似度的函数。
            output_dir (str): 图表保存的输出目录（可选）。
        """
        print("生成评估报告...")
        runtime_comparison = self.compute_runtime_comparison()
        near_duplicate_rate = self.compute_near_duplicate_rate(similarity_func)

        print("\n评估报告:")
        print(f"候选对总数: {len(self.candidate_pairs)}")
        print(f"近重复文档比率: {near_duplicate_rate:.2f}")
        print("运行时性能对比:")
        for module, runtime in runtime_comparison.items():
            print(f"  {module}: {runtime:.2f} 秒")

        self.generate_visualizations(output_dir)


# 示例用法
if __name__ == "__main__":
    # 示例数据集
    data = ["document1", "document2", "document3", "document4"]

    # 示例候选对
    candidate_pairs = [(0, 1), (1, 2), (2, 3)]

    # 示例运行时日志
    runtime_log = {
        "模块1": 1.23,
        "模块2": 0.98,
        "模块3": 2.45
    }

    # 示例相似度函数
    def dummy_similarity(doc1, doc2):
        return 0.8 if doc1 != doc2 else 1.0

    # 初始化评估器
    evaluator = Evaluator(candidate_pairs, runtime_log, data)

    # 生成评估报告
    evaluator.generate_report(dummy_similarity, threshold=0.75)
