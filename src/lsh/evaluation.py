'''
	•	功能：对LSH候选对以及整个去重系统进行评估，计算去重率、查全率、查准率以及运行时间等指标。
	•	主要类与函数：
	•	Class Evaluator
	•	__init__(self, candidate_pairs, ground_truth=None): 初始化候选对和（如果有的话）真实标签数据。
	•	compute_duplicate_rate(self): 计算近重复文档比率。
	•	compute_performance_metrics(self): 计算评估指标，如运行时间、误报率等。
	•	generate_report(self): 输出统计结果，并生成可视化图表（如调用matplotlib绘制图像）。
	•	调用：在主流程中，通过main.py调用该评估模块，对LSH得到的候选近重复对进行性能和质量评估。
'''
from typing import Set, Tuple, Optional
import time
import matplotlib.pyplot as plt


class Evaluator:
    """
    Evaluator 类，用于对 LSH 候选对以及整个去重系统进行评估。
    """

    def __init__(self, candidate_pairs: Set[Tuple[int, int]], ground_truth: Optional[Set[Tuple[int, int]]] = None):
        """
        初始化 Evaluator 实例。

        参数:
            candidate_pairs (Set[Tuple[int, int]]): LSH 生成的候选近重复文档对。
            ground_truth (Optional[Set[Tuple[int, int]]]): 真实的近重复文档对（如果有）。
        """
        self.candidate_pairs = candidate_pairs
        self.ground_truth = ground_truth

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

    def compute_performance_metrics(self) -> dict:
        """
        计算评估指标，包括查全率、查准率和 F1 分数。

        返回:
            dict: 包含查全率、查准率和 F1 分数的字典。
        """
        if self.ground_truth is None:
            raise ValueError("需要提供 ground_truth 才能计算性能指标。")

        true_positives = len(self.candidate_pairs & self.ground_truth)
        false_positives = len(self.candidate_pairs - self.ground_truth)
        false_negatives = len(self.ground_truth - self.candidate_pairs)

        precision = true_positives / \
            (true_positives + false_positives) if (true_positives +
                                                   false_positives) > 0 else 0.0
        recall = true_positives / \
            (true_positives + false_negatives) if (true_positives +
                                                   false_negatives) > 0 else 0.0
        f1_score = (2 * precision * recall) / (precision +
                                               recall) if (precision + recall) > 0 else 0.0

        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
        }

    def generate_report(self, metrics: dict, runtime: float, output_dir: str = None) -> None:
        """
        输出统计结果，并生成可视化图表。

        参数:
            metrics (dict): 性能指标字典。
            runtime (float): 运行时间。
        """
        print("评估报告:")
        print(f"运行时间: {runtime:.2f} 秒")
        print(f"查准率 (Precision): {metrics['precision']:.2f}")
        print(f"查全率 (Recall): {metrics['recall']:.2f}")
        print(f"F1 分数: {metrics['f1_score']:.2f}")
        print(f"True Positives: {metrics['true_positives']}")
        print(f"False Positives: {metrics['false_positives']}")
        print(f"False Negatives: {metrics['false_negatives']}")

        # 可视化查准率和查全率
        labels = ["Precision", "Recall", "F1 Score"]
        values = [metrics["precision"], metrics["recall"], metrics["f1_score"]]

        plt.bar(labels, values, color=["blue", "green", "orange"])
        plt.ylim(0, 1)
        plt.title("Performance Metrics")
        plt.ylabel("Score")
        plt.show()

        # 保存图表到文件
        if output_dir:
            plt.savefig(f"{output_dir}/performance_metrics.png")
            print(f"图表已保存到: {output_dir}/performance_metrics.png")
        else:
            print("未指定输出目录，图表未保存。")


# 示例用法
if __name__ == "__main__":
    # 示例候选对和真实标签
    candidate_pairs = {(1, 2), (2, 3), (4, 5)}
    ground_truth = {(1, 2), (3, 4), (4, 5)}

    # 初始化评估器
    evaluator = Evaluator(candidate_pairs, ground_truth)

    # 计算运行时间
    start_time = time.time()
    metrics = evaluator.compute_performance_metrics()
    runtime = time.time() - start_time

    # 生成报告
    evaluator.generate_report(metrics, runtime)
