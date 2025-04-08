# 食用方法
1. 修改*config/config.yaml*下的配置
2. 运行*python src/main.py*
# 运行逻辑
1. 导入配置：加载config/config.yaml。
2. 初始化日志：调用utils/logger.py中的设置函数。
3. 数据加载：调用utils/data_loader.py读取原始数据。
4. 预处理：调用preprocessing.py中预处理类或函数，对原始数据进行清洗、token化以及n-gram生成。
5. 特征提取：调用feature_extraction.py，生成文本特征表示（如set或向量）。
6. 指纹生成：根据配置依次调用fingerprint/minhash.py、fingerprint/simhash.py、fingerprint/bitsampling.py中的类与方法生成文本指纹。
7. LSH索引构建：传入各类签名到lsh/lsh_index.py中，建立LSH索引，过滤产生候选文档对。
8. 评估：调用lsh/evaluation.py对生成的候选对进行定量与定性评估。
9. 输出结果：将评估报告、日志和候选对结果写入data/results/目录中。
