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

class DataLoader:
    """
    数据加载与保存工具类。
    提供从文件加载数据和将数据保存到文件的功能。
    """

    def load_data(self, file_path: str) -> list[str]:
        """
        从指定路径加载原始数据。
        
        参数:
            file_path (str): 数据文件路径（支持 Parquet 格式）。
        
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
            
            # 检查是否包含文本列
            if "text" not in df.columns:
                raise ValueError("数据文件中缺少 'text' 列。")
            
            # 返回原始文本数据列表
            return df["text"].tolist()
        else:
            raise ValueError(f"不支持的文件类型: {file_extension}")

    def save_data(self, data, output_path: str):
        """
        将数据保存到指定路径。
        （此方法暂未实现）
        """
        pass

# 示例用法
if __name__ == "__main__":
    data_loader = DataLoader()
    
    # 加载数据示例
    try:
        raw_data = data_loader.load_data("data/raw/sample_test.parquet")
        print("加载的原始数据:", raw_data[:5])  # 打印前 5 条数据
    except Exception as e:
        print(e)