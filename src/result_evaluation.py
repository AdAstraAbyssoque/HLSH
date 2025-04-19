import argparse
import random
from utils.data_loader import DataLoader
from termcolor import colored
import yaml

def visualize_diff_terminal(text1: str, text2: str) -> None:
    """
    在终端中以高亮方式可视化两个文段的差异。
    """
    words1 = text1.split()
    words2 = text2.split()
    
    # 使用动态规划计算最长公共子序列（LCS）
    dp = [[0] * (len(words2) + 1) for _ in range(len(words1) + 1)]
    for i in range(1, len(words1) + 1):
        for j in range(1, len(words2) + 1):
            if words1[i - 1] == words2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    
    # 回溯找到 LCS
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
    
    # 高亮显示差异
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
                result1.append(colored(words1[i], "red"))  # 删除的部分标红
                i += 1
            if j < len(words2) and (k >= len(lcs) or words2[j] != lcs[k]):
                result2.append(colored(words2[j], "green"))  # 新增的部分标绿
                j += 1
    
    # 输出结果
    print(" ".join(result1))
    print()
    print(" ".join(result2))

def visualize_diff_latex(text1: str, text2: str) -> str:
    """
    生成两个文段的差异，并以 LaTeX 格式高亮显示。
    
    参数:
        text1 (str): 文本1。
        text2 (str): 文本2。
    
    返回:
        str: 包含 LaTeX 格式的差异文本。
    """
    words1 = text1.split()
    words2 = text2.split()
    
    # 使用动态规划计算最长公共子序列（LCS）
    dp = [[0] * (len(words2) + 1) for _ in range(len(words1) + 1)]
    for i in range(1, len(words1) + 1):
        for j in range(1, len(words2) + 1):
            if words1[i - 1] == words2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    
    # 回溯找到 LCS
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
    
    # 高亮显示差异
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
                result1.append(f"\\textcolor{{red}}{{{words1[i]}}}")  # 删除的部分标红
                i += 1
            if j < len(words2) and (k >= len(lcs) or words2[j] != lcs[k]):
                result2.append(f"\\textcolor{{green}}{{{words2[j]}}}")  # 新增的部分标绿
                j += 1
    
    # 合并结果
    latex_result1 = " ".join(result1)
    latex_result2 = " ".join(result2)
    
    # 返回 LaTeX 格式的文本
    return f"文段1（删除的部分标红）:\n\n{latex_result1}\n\n文段2（新增的部分标绿）:\n\n{latex_result2}"

def visualize_diff_html(pairs: list[tuple], data: list[str], output_path: str) -> None:
    """
    生成多个候选对的差异，并以 HTML 格式高亮显示，保存到指定路径。
    
    参数:
        pairs (list[tuple]): 候选对列表，每个元素为 (id1, id2)。
        data (list[str]): 文本数据列表。
        output_path (str): HTML 文件保存路径。
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
        <h1>候选对差异报告</h1>
    """

    for idx, (id1, id2) in enumerate(pairs):
        text1 = data[id1]
        text2 = data[id2]

        # 使用动态规划计算最长公共子序列（LCS）
        words1 = text1.split()
        words2 = text2.split()
        dp = [[0] * (len(words2) + 1) for _ in range(len(words1) + 1)]
        for i in range(1, len(words1) + 1):
            for j in range(1, len(words2) + 1):
                if words1[i - 1] == words2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        # 回溯找到 LCS
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

        # 高亮显示差异
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

        # 合并结果
        html_result1 = " ".join(result1)
        html_result2 = " ".join(result2)

        # 添加到 HTML 内容
        html_content += f"""
        <div class="pair">
            <h2>候选对 {idx + 1}: 文档 {id1} vs 文档 {id2}</h2>
            <h3>文段1（删除的部分标红）:</h3>
            <p>{html_result1}</p>
            <h3>文段2（新增的部分标绿）:</h3>
            <p>{html_result2}</p>
        </div>
        """

    html_content += """
    </body>
    </html>
    """

    # 保存 HTML 文件
    with open(output_path, "w", encoding="utf-8") as file:
        file.write(html_content)
    print(f"HTML 差异报告已保存到: {output_path}")

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


if __name__ == "__main__":
    # 使用 argparse 解析命令行参数
    parser = argparse.ArgumentParser(description="Evaluate and visualize text differences.")
    parser.add_argument("random_pairs_num", type=int, help="Number of random candidate pairs to visualize.")
    args = parser.parse_args()

    # 设置路径
    config_path = "config/config.yaml"
    config = load_config(config_path)
    rawdata_path = config["data"]["raw_data_path"]
    final_result_path = config["output"]["results_path"]
    evaluation_html_path = config["output"]["evaluation_html_path"]
    
    # 加载数据
    data_loader = DataLoader()
    data = data_loader.load_data(rawdata_path, parallel_enabled=True)
    candidate_pairs = data_loader.load_candidate_pairs_csv(final_result_path)
    
    # 随机选择候选对
    random_pairs_num = args.random_pairs_num
    random_candidate_pairs = random.sample(candidate_pairs, random_pairs_num)
    
    # 生成 HTML 差异报告
    visualize_diff_html(random_candidate_pairs, data, evaluation_html_path)
    # 可视化差异
    for idx, pair in enumerate(random_candidate_pairs):
        print(f"Pair {idx + 1}:")
        visualize_diff_terminal(data[pair[0]], data[pair[1]])
        print("=" * 60)