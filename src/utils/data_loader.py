'''
	•	功能：提供数据的加载与保存接口。
	•	主要类与函数：
	•	Class DataLoader
	•	def load_data(self, file_path): 读取原始文本数据（例如从文件或数据库中）。
	•	def save_data(self, data, output_path): 将预处理结果或评估结果写入指定路径。
	•	调用：main.py最开始调用DataLoader加载Wiki40B数据。
'''
import os
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from tqdm import tqdm


class DataLoader:
    """
    数据加载与保存工具类。
    提供从文件加载数据和将数据保存到文件的功能。
    """

    def save_preprocessed_data(self, data: list[str], output_path: str):
        """
        将预处理后的数据保存到指定路径。

        参数:
            data (list[str]): 预处理后的文本数据列表。
            output_path (str): 保存文件的路径（支持 Parquet 和 CSV 格式）。

        返回:
            None
        """
        if not data:
            raise ValueError("数据为空，无法保存。")

        _, file_extension = os.path.splitext(output_path)

        # 将数据转换为 DataFrame
        df = pd.DataFrame(data, columns=["text"])

        if file_extension == ".parquet":
            try:
                df.to_parquet(output_path, index=False)
                print(f"预处理数据已成功保存为 Parquet 文件: {output_path}")
            except Exception as e:
                raise ValueError(f"无法保存为 Parquet 文件: {e}")
        elif file_extension == ".csv":
            try:
                df.to_csv(output_path, index=False)
                print(f"预处理数据已成功保存为 CSV 文件: {output_path}")
            except Exception as e:
                raise ValueError(f"无法保存为 CSV 文件: {e}")
        else:
            raise ValueError(f"不支持的文件类型: {file_extension}")

    def load_data(self, file_path: str, parallel_enabled: bool, thread_pool_size=4) -> list[str]:
        """
        从指定路径加载原始数据, 且可区分是否为目录路径或单个文件路径。
        如果是目录路径，则加载该目录下所有的 Parquet 和 CSV 文件。
        如果是单个文件路径，则加载该文件。
        如果文件不存在或格式不支持，则抛出异常。

        参数:
            file_path (str): 数据文件路径（支持 Parquet 和 CSV 格式）。

        返回:
            list[str]: 原始文本数据列表。
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"路径不存在: {file_path}")
        elif os.path.isfile(file_path):
            return self._load_data(file_path)
        elif os.path.isdir(file_path):
            data = []
            if parallel_enabled:
                with ThreadPoolExecutor(max_workers=thread_pool_size) as executor:
                    futures = []
                    for root, _, files in os.walk(file_path):
                        for file in files:
                            file_path = os.path.join(root, file)
                            if file_path.endswith((".parquet", ".csv")):
                                futures.append(executor.submit(
                                    self._load_data, file_path))
                    for future in futures:
                        try:
                            data.extend(future.result())
                        except Exception as e:
                            print(f"加载文件时发生错误: {e}")
            else:
                for root, _, files in os.walk(file_path):
                    for file in tqdm(files, desc="加载文件"):
                        file_path = os.path.join(root, file)
                        try:
                            data.extend(self._load_data(file_path))
                        except Exception as e:
                            print(f"加载文件 {file_path} 时发生错误: {e}")
            return data
        else:
            raise ValueError(f"无效的文件路径: {file_path}")

    def _load_data(self, file_path: str) -> list[str]:
        """
        从指定路径加载原始数据。

        参数:
            file_path (str): 数据文件路径（支持 Parquet 和 CSV 格式）。

        返回:
            list[str]: 原始文本数据列表。
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件未找到: {file_path}")

        _, file_extension = os.path.splitext(file_path)

        if file_extension == ".parquet":
            # 读取 Parquet 文件
            try:
                df = pd.read_parquet(file_path)
            except Exception as e:
                raise ValueError(f"无法读取 Parquet 文件: {e}")

        elif file_extension == ".csv":
            # 读取 CSV 文件
            try:
                df = pd.read_csv(file_path)
            except Exception as e:
                raise ValueError(f"无法读取 CSV 文件: {e}")

        else:
            raise ValueError(f"不支持的文件类型: {file_extension}")

        # 检查是否包含文本列
        if "text" not in df.columns:
            raise ValueError("数据文件中缺少 'text' 列。")

        # 返回原始文本数据列表
        return df["text"].tolist()

    def save_signatures(self, data, output_path: str):
        """
        将数据保存到指定路径。

        参数:
            data (list): 要保存的数据列表（可以是签名指纹或文本数据）。
            output_path (str): 保存文件的路径（支持 Parquet 和 CSV 格式）。

        返回:
            None
        """
        if not data:
            raise ValueError("数据为空，无法保存。")

        _, file_extension = os.path.splitext(output_path)

        # 检查数据类型并转换为 DataFrame
        if isinstance(data[0], list) or isinstance(data[0], tuple):
            # 如果是签名指纹（嵌套列表或元组）
            df = pd.DataFrame(data)
        else:
            # 如果是普通文本数据
            df = pd.DataFrame(data, columns=["text"])

        if file_extension == ".parquet":
            try:
                df.to_parquet(output_path, index=False)
                print(f"数据已成功保存为 Parquet 文件: {output_path}")
            except Exception as e:
                raise ValueError(f"无法保存为 Parquet 文件: {e}")
        elif file_extension == ".csv":
            try:
                df.to_csv(output_path, index=False)
                print(f"数据已成功保存为 CSV 文件: {output_path}")
            except Exception as e:
                raise ValueError(f"无法保存为 CSV 文件: {e}")
        else:
            raise ValueError(f"不支持的文件类型: {file_extension}")


# 示例用法
if __name__ == "__main__":
    data_loader = DataLoader()

    loader_filepath = "data/raw/test"
    # load data
    data = data_loader.load_data(loader_filepath, True, 4)
    print("加载的数据:", data[:10])

    # 示例签名指纹数据
    signatures = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9]
    ]

    # 保存签名指纹
    try:
        data_loader.save_signatures(
            signatures, "data/processed/signatures.parquet")
        print("签名指纹已保存。")
    except Exception as e:
        print(e)
