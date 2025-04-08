import pandas as pd

# 构造一个简单的 DataFrame
data = {
    "id": [1, 2, 3, 4, 5, 6, 7, 8],
    "text": [
        "The quick brown fox jumps over the lazy dog",
        "A journey of a thousand miles begins with a single step",
        "To be or not to be, that is the question",
        "All that glitters is not gold",
        "The early bird catches the worm",
        "Actions speak louder than words",
        "A picture is worth a thousand words",
        "A picture is worth a words"
    ]
}

df = pd.DataFrame(data)

# 保存为 Parquet 文件（需要安装 pyarrow 或 fastparquet）
df.to_parquet("sample_test.parquet", index=False)
