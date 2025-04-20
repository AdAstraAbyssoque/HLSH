import time
from typing import List, Dict, Tuple, Callable
import matplotlib.pyplot as plt
from tqdm import tqdm

class LSHEvaluator:
    """
    LSHEvaluator: evaluates an LSH system’s performance, including
    runtime profiling and near-duplicate document rate calculation.
    """

    def __init__(
        self,
        candidate_pairs: List[Tuple[int, int]],
        runtime_log: Dict[str, float],
        data: List[str],
    ):
        """
        Initialize the evaluator.

        Parameters:
            candidate_pairs: list of index pairs produced by LSH
            runtime_log: mapping from module/stage names to their execution time
            data: the original list of documents
        """
        self.candidate_pairs = candidate_pairs
        self.runtime_log = runtime_log
        self.data = data

    def compute_runtime_comparison(self) -> Dict[str, float]:
        """
        Print and return runtime per module/stage.

        Returns:
            A dict mapping each module name to its elapsed time (seconds).
        """
        print("Runtime Performance Comparison:")
        for module, elapsed in self.runtime_log.items():
            print(f"  {module}: {elapsed:.2f} seconds")
        return self.runtime_log

    def compute_near_duplicate_rate(self, similarity_func: Callable[[str, str], float]) -> float:
        """
        Compute the average near-duplicate similarity over all candidate pairs.

        Parameters:
            similarity_func: a function that takes two documents (strings)
                             and returns a similarity score (0.0–1.0).

        Returns:
            The average similarity (near-duplicate rate) across all pairs.
        """
        total_similarity = 0.0
        for idx1, idx2 in tqdm(self.candidate_pairs, desc="Calculating near-duplicate rate"):
            total_similarity += similarity_func(self.data[idx1], self.data[idx2])

        if not self.candidate_pairs:
            return 0.0
        return total_similarity / len(self.candidate_pairs)

    def generate_visualizations(self, output_dir: str = None) -> None:
        """
        Create and optionally save charts with matplotlib.

        Parameters:
            output_dir: if provided, directory where plots are saved as PNGs;
                        otherwise, plots are shown interactively.
        """
        # Bar chart of runtimes
        modules = list(self.runtime_log.keys())
        times = list(self.runtime_log.values())

        plt.figure(figsize=(10, 5))
        plt.bar(modules, times)
        plt.xlabel("Module")
        plt.ylabel("Time (seconds)")
        plt.title("Runtime Performance Comparison")
        plt.xticks(rotation=45)
        plt.tight_layout()

        if output_dir:
            path = f"{output_dir}/runtime_comparison.png"
            plt.savefig(path)
            print(f"Saved runtime comparison chart to {path}")
        else:
            plt.show()

    def generate_report(
        self,
        similarity_func: Callable[[str, str], float],
        output_dir: str = None
    ) -> None:
        """
        Run all evaluations, print a summary report, and produce visualizations.

        Parameters:
            similarity_func: function to compute similarity between document pairs
            output_dir: directory to save charts (optional)
        """
        print("Generating evaluation report...\n")
        runtime_data = self.compute_runtime_comparison()
        near_dup_rate = self.compute_near_duplicate_rate(similarity_func)

        print("\n=== Evaluation Report ===")
        print(f"Total candidate pairs: {len(self.candidate_pairs)}")
        print(f"Average near-duplicate rate: {near_dup_rate:.2f}")
        print("\nModule runtimes:")
        for module, elapsed in runtime_data.items():
            print(f"  - {module}: {elapsed:.2f} seconds")

        self.generate_visualizations(output_dir)


# Example usage
if __name__ == "__main__":
    # Sample dataset
    documents = ["doc1 text...", "doc2 text...", "doc3 text...", "doc4 text..."]

    # Sample candidate pairs from LSH
    candidate_pairs = [(0, 1), (1, 2), (2, 3)]

    # Sample runtime log
    runtime_log = {
        "Shingling": 1.23,
        "MinHash": 0.98,
        "LSH Bucketing": 2.45
    }

    # Example similarity function
    def dummy_similarity(doc_a: str, doc_b: str) -> float:
        return 1.0 if doc_a == doc_b else 0.8

    evaluator = LSHEvaluator(candidate_pairs, runtime_log, documents)
    evaluator.generate_report(dummy_similarity, output_dir=None)