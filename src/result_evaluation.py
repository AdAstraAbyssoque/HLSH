import argparse
import random
from utils.data_loader import DataLoader
from termcolor import colored
import yaml

def visualize_diff_terminal(text1: str, text2: str) -> None:
    """
    Visualize the differences between two texts in the terminal with highlights.
    """
    words1 = text1.split()
    words2 = text2.split()
    
    # Use dynamic programming to calculate the Longest Common Subsequence (LCS)
    dp = [[0] * (len(words2) + 1) for _ in range(len(words1) + 1)]
    for i in range(1, len(words1) + 1):
        for j in range(1, len(words2) + 1):
            if words1[i - 1] == words2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    
    # Backtrack to find the LCS
    i, j = len(words1), len(words2)
    lcs = []
    while i > 0 and j > 0:
        if words1[i - 1] == words2[j - 1]:
            lcs.append(words1[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    lcs.reverse()
    
    # Highlight differences
    result1 = []
    result2 = []
    i, j, k = 0, 0, 0
    while i < len(words1) or j < len(words2):
        if k < len(lcs) and i < len(words1) and j < len(words2) and words1[i] == words2[j] == lcs[k]:
            result1.append(words1[i])
            result2.append(words2[j])
            i += 1
            j += 1
            k += 1
        else:
            if i < len(words1) and (k >= len(lcs) or words1[i] != lcs[k]):
                result1.append(colored(words1[i], "red"))  # Deleted parts in red
                i += 1
            if j < len(words2) and (k >= len(lcs) or words2[j] != lcs[k]):
                result2.append(colored(words2[j], "green"))  # Added parts in green
                j += 1
    
    # Print results
    print(" ".join(result1))
    print()
    print(" ".join(result2))

def visualize_diff_latex(text1: str, text2: str) -> str:
    """
    Generate the differences between two texts and highlight them in LaTeX format.
    
    Parameters:
        text1 (str): Text 1.
        text2 (str): Text 2.
    
    Returns:
        str: Text with differences highlighted in LaTeX format.
    """
    words1 = text1.split()
    words2 = text2.split()
    
    # Use dynamic programming to calculate the Longest Common Subsequence (LCS)
    dp = [[0] * (len(words2) + 1) for _ in range(len(words1) + 1)]
    for i in range(1, len(words1) + 1):
        for j in range(1, len(words2) + 1):
            if words1[i - 1] == words2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    
    # Backtrack to find the LCS
    i, j = len(words1), len(words2)
    lcs = []
    while i > 0 and j > 0:
        if words1[i - 1] == words2[j - 1]:
            lcs.append(words1[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    lcs.reverse()
    
    # Highlight differences
    result1 = []
    result2 = []
    i, j, k = 0, 0, 0
    while i < len(words1) or j < len(words2):
        if k < len(lcs) and i < len(words1) and j < len(words2) and words1[i] == words2[j] == lcs[k]:
            result1.append(words1[i])
            result2.append(words2[j])
            i += 1
            j += 1
            k += 1
        else:
            if i < len(words1) and (k >= len(lcs) or words1[i] != lcs[k]):
                result1.append(f"\\textcolor{{red}}{{{words1[i]}}}")  # Deleted parts in red
                i += 1
            if j < len(words2) and (k >= len(lcs) or words2[j] != lcs[k]):
                result2.append(f"\\textcolor{{green}}{{{words2[j]}}}")  # Added parts in green
                j += 1
    
    # Combine results
    latex_result1 = " ".join(result1)
    latex_result2 = " ".join(result2)
    
    # Return LaTeX formatted text
    return f"Text 1 (deleted parts in red):\n\n{latex_result1}\n\nText 2 (added parts in green):\n\n{latex_result2}"

def visualize_diff_html(pairs: list[tuple], data: list[str], output_path: str) -> None:
    """
    Generate differences for multiple candidate pairs and highlight them in HTML format, saving to the specified path.
    
    Parameters:
        pairs (list[tuple]): List of candidate pairs, each element is (id1, id2).
        data (list[str]): List of text data.
        output_path (str): Path to save the HTML file.
    """
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Text Differences</title>
        <style>
            body { font-family: Arial, sans-serif; line-height: 1.6; margin: 20px; }
            .pair { margin-bottom: 40px; }
            .pair h2 { margin-bottom: 10px; }
            .deleted { color: red; }
            .added { color: green; }
        </style>
    </head>
    <body>
        <h1>Candidate Pair Difference Report</h1>
    """

    for idx, (id1, id2) in enumerate(pairs):
        text1 = data[id1]
        text2 = data[id2]

        # Use dynamic programming to calculate the Longest Common Subsequence (LCS)
        words1 = text1.split()
        words2 = text2.split()
        dp = [[0] * (len(words2) + 1) for _ in range(len(words1) + 1)]
        for i in range(1, len(words1) + 1):
            for j in range(1, len(words2) + 1):
                if words1[i - 1] == words2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        # Backtrack to find the LCS
        i, j = len(words1), len(words2)
        lcs = []
        while i > 0 and j > 0:
            if words1[i - 1] == words2[j - 1]:
                lcs.append(words1[i - 1])
                i -= 1
                j -= 1
            elif dp[i - 1][j] >= dp[i][j - 1]:
                i -= 1
            else:
                j -= 1
        lcs.reverse()

        # Highlight differences
        result1 = []
        result2 = []
        i, j, k = 0, 0, 0
        while i < len(words1) or j < len(words2):
            if k < len(lcs) and i < len(words1) and j < len(words2) and words1[i] == words2[j] == lcs[k]:
                result1.append(words1[i])
                result2.append(words2[j])
                i += 1
                j += 1
                k += 1
            else:
                if i < len(words1) and (k >= len(lcs) or words1[i] != lcs[k]):
                    result1.append(f'<span class="deleted">{words1[i]}</span>')
                    i += 1
                if j < len(words2) and (k >= len(lcs) or words2[j] != lcs[k]):
                    result2.append(f'<span class="added">{words2[j]}</span>')
                    j += 1

        # Combine results
        html_result1 = " ".join(result1)
        html_result2 = " ".join(result2)

        # Add to HTML content
        html_content += f"""
        <div class="pair">
            <h2>Candidate Pair {idx + 1}: Document {id1} vs Document {id2}</h2>
            <p>{html_result1}</p>
            <p>{html_result2}</p>
        </div>
        """

    html_content += """
    </body>
    </html>
    """

    # Save HTML file
    with open(output_path, "w", encoding="utf-8") as file:
        file.write(html_content)
    print(f"HTML difference report saved to: {output_path}")

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


if __name__ == "__main__":
    # Use argparse to parse command-line arguments
    parser = argparse.ArgumentParser(description="Evaluate and visualize text differences.")
    parser.add_argument("random_pairs_num", type=int, help="Number of random candidate pairs to visualize.")
    args = parser.parse_args()

    # Set paths
    config_path = "config/config.yaml"
    config = load_config(config_path)
    rawdata_path = config["data"]["raw_data_path"]
    final_result_path = config["output"]["results_path"]
    evaluation_html_path = config["output"]["evaluation_html_path"]
    
    # Load data
    data_loader = DataLoader()
    data = data_loader.load_data(rawdata_path, parallel_enabled=True)
    candidate_pairs = data_loader.load_candidate_pairs_csv(final_result_path)
    
    # Randomly select candidate pairs
    random_pairs_num = args.random_pairs_num
    random_candidate_pairs = random.sample(candidate_pairs, random_pairs_num)
    
    # Generate HTML difference report
    visualize_diff_html(random_candidate_pairs, data, evaluation_html_path)
    # Visualize differences in the terminal
    for idx, pair in enumerate(random_candidate_pairs):
        print(f"Pair {idx + 1}:")
        visualize_diff_terminal(data[pair[0]], data[pair[1]])
        print("=" * 60)