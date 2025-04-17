'''
	•	作用：作为整个系统的入口文件，负责整个工作流程的调度。
	•	主要内容：
	•	导入配置：加载config/config.yaml。
	•	初始化日志：调用utils/logger.py中的设置函数。
	•	数据加载：调用utils/data_loader.py读取原始数据。
	•	预处理：调用preprocessing.py中预处理类或函数，对原始数据进行清洗、token化以及n-gram生成。
	•	特征提取：调用feature_extraction.py，生成文本特征表示（如set或向量）。
	•	指纹生成：根据配置依次调用fingerprint/minhash.py、fingerprint/simhash.py、fingerprint/bitsampling.py中的类与方法生成文本指纹。
	•	LSH索引构建：传入各类签名到lsh/lsh_index.py中，建立LSH索引，过滤产生候选文档对。
	•	评估：调用lsh/evaluation.py或evaluation.py对生成的候选对进行定量与定性评估。
	•	输出结果：将评估报告、日志和候选对结果写入data/results/目录中。
	•	调用关系：是其他所有模块的上层调用入口，通过依次调用预处理、特征提取、指纹生成、LSH索引和评估模块，形成整体流水线。
'''
import os
import yaml
import time
import gc
from sklearn.feature_extraction.text import TfidfVectorizer
from utils.logger import setup_logger, Log_pipeline_info
from utils.data_loader import DataLoader
from feature_extraction import FeatureExtractor
from fingerprint.minhash import MinHash
from fingerprint.simhash import SimHash
from fingerprint.bitsampling import BitSampling
from lsh.lsh_index import MinHashLSHIndex, SimHashLSHIndex, BitSamplingLSHIndex, HybridLSHIndex
from lsh.evaluation import Evaluator
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from lsh.helper import cosine_similarity, jaccard_similarity, euclidean_distance
from preprocessing import preprocess_text, parallel_preprocess_texts


# 加载配置文件


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def main():
    # 1. 加载配置
    config_path = "config/config.yaml"
    config = load_config(config_path)

    # 1.1 获取并行配置
    parallel_config = config.get("parallel", {})
    parallel_enabled = parallel_config.get("enabled", False)
    thread_pool_size = parallel_config.get("thread_pool_size", 4)
    process_pool_size = parallel_config.get("process_pool_size", 8)

    # 2. 初始化日志和时间记录
    log_file = config["logging"]["log_file"]
    log_level = config["logging"]["log_level"]
    logger = setup_logger(log_file, log_level)
    logger.info("-------------------------------------------------------")
    logger.info("系统启动，加载配置完成。")

    # pipeline_log 初始化参数
    pipeline_log = Log_pipeline_info(config)

    # 3. 数据加载
    data_loader_time = time.time()
    data_loader = DataLoader()
    raw_data_path = config["data"]["raw_data_path"]  # 可以是文件路径或目录路径
    logger.info(f"加载数据路径：{raw_data_path}")

    raw_data = []
    data_loader_parallel = parallel_config.get("data_loader_parallel", False)
    raw_data = data_loader.load_data(
        raw_data_path, parallel_enabled=data_loader_parallel, thread_pool_size=thread_pool_size)

    logger.info(f"数据加载完成，共加载 {len(raw_data)} 条记录。")
    pipeline_log.add_result("raw_data_count", len(raw_data))
    data_loader_time = time.time() - data_loader_time
    pipeline_log.add_runtime("data_loader_time", data_loader_time)
    logger.info(f"数据加载时间：{data_loader_time:.2f} 秒")

    # raw_data=raw_data[:4000]  # 测试时只取前1000条数据

    # 4. 数据预处理
    preprocess_data_time = time.time()
    logger.info("开始数据预处理...")

    preprocess_parallel = parallel_config.get("preprocess_parallel", True)
    preprocessed_data = parallel_preprocess_texts(
        raw_data, process_pool_size=process_pool_size, parallel_enabled=preprocess_parallel
    )

    logger.info("数据预处理完成。")
    preprocess_data_time = time.time() - preprocess_data_time
    pipeline_log.add_runtime("preprocess_data_time", preprocess_data_time)
    logger.info(f"数据预处理时间：{preprocess_data_time:.2f} 秒")

    # 5. 特征提取
    feature_extraction_time = time.time()
    feature_method = config["feature_extraction"]["method"]
    ngram_size = config["feature_extraction"].get("ngram_size", 3)

    logger.info(f"开始特征提取，方法：{feature_method}")
    extractor = FeatureExtractor(method=feature_method, n=ngram_size)
    feature_extraction_parallel = parallel_config.get(
        "feature_extraction_parallel", True)
    logger.info("开启并行" if feature_extraction_parallel else "关闭并行")

    # 调用 FeatureExtractor 的并行特征提取方法
    features = extractor.parallel_extract_features(
        preprocessed_data, process_pool_size=process_pool_size, parallel_enabled=feature_extraction_parallel
    )

    idf = None
    if feature_method == "vectorize":
        token_for_simhash = [extractor._extract_ngrams(
            text) for text in preprocessed_data]
        idf, features = features, token_for_simhash

    logger.info("特征提取完成。")
    feature_extraction_time = time.time() - feature_extraction_time
    pipeline_log.add_runtime("feature_extraction_time",
                             feature_extraction_time)
    logger.info(f"特征提取时间：{feature_extraction_time:.2f} 秒")


    # 6. 指纹生成
    fingerprint_time = time.time()
    fingerprint_method = config["fingerprint"]["method"]
    use_cache = parallel_config.get("use_memory_cache", True)
    logger.info(f"开始指纹生成，方法：{fingerprint_method}")
    fingerprint_parallel = parallel_config.get("fingerprint_parallel", True)
    logger.info("开启并行" if fingerprint_parallel else "关闭并行")

    if fingerprint_method == "hybrid":
        # 混合方法需要同时生成minhash和simhash签名
        num_hashes = config["fingerprint"]["num_hashes"]
        hash_bits = config["fingerprint"]["hash_bits"]
        seed = config["fingerprint"].get("seed", None)

        minhash = MinHash(num_hashes=num_hashes, seed=seed)
        simhash = SimHash(hash_bits=hash_bits)

        # 生成minhash签名
        minhash_signatures = minhash.parallel_compute_signature(
            features, parallel_enable=fingerprint_parallel, process_pool_size=process_pool_size
        )
        # 生成simhash签名
        simhash_signatures = simhash.parallel_compute_signature(
            features, idf=idf, parallel_enable=fingerprint_parallel, process_pool_size=process_pool_size
        )
        # 合并签名
        signatures = []
        for minhash_sig, simhash_sig in zip(minhash_signatures, simhash_signatures):
            combined_sig = {
                "minhash": minhash_sig,
                "simhash": simhash_sig
            }
            signatures.append(combined_sig)

    elif fingerprint_method == "minhash":
        num_hashes = config["fingerprint"]["num_hashes"]
        seed = config["fingerprint"].get("seed", None)
        minhash = MinHash(num_hashes=num_hashes, seed=seed)
        signatures = minhash.parallel_compute_signature(
            features, parallel_enable=parallel_enabled, process_pool_size=process_pool_size
        )
    elif fingerprint_method == "simhash":
        hash_bits = config["fingerprint"]["hash_bits"]
        simhash = SimHash(hash_bits=hash_bits)
        signatures = simhash.parallel_compute_signature(
            features, idf=idf, parallel_enable=parallel_enabled, process_pool_size=process_pool_size
        )

    # 如果启用内存缓存，清理进程池
    if use_cache:
        gc.collect()

    logger.info("指纹生成完成，共生成签名数量：{}".format(len(signatures)))
    pipeline_log.add_result("signature_count", len(signatures))
    fingerprint_time = time.time() - fingerprint_time
    pipeline_log.add_runtime("fingerprint_time", fingerprint_time)
    logger.info(f"指纹生成时间：{fingerprint_time:.2f} 秒")

    # 使用 DataLoader 保存签名指纹
    fingerprint_output_path = config["output"]["fingerpritnts_path"]
    try:
        logger.info(f"开始保存签名指纹到文件：{fingerprint_output_path}")
        data_loader.save_signatures(signatures, fingerprint_output_path)
        logger.info(f"签名指纹已成功保存到文件：{fingerprint_output_path}")
    except Exception as e:
        logger.error(f"保存签名指纹失败，错误信息：{e}")
        return
    
    # 7. LSH 索引构建
    lsh_index_time = time.time()
    lsh_method = config["lsh"]["method"]
    logger.info(f"开始 LSH 索引构建，方法：{lsh_method}")

    if lsh_method == "minhash":
        num_bands = config["lsh"]["num_bands"]
        rows_per_band = config["lsh"]["rows_per_band"]
        lsh_index = MinHashLSHIndex(
            num_bands=num_bands, rows_per_band=rows_per_band)

    elif lsh_method == "simhash":
        radius = config["lsh"]["radius"]
        hash_bits = config["fingerprint"]["hash_bits"]
        lsh_index = SimHashLSHIndex(radius=radius, hash_bits=hash_bits)

    elif lsh_method == "bitsampling":
        num_hash_tables = config["lsh"]["num_hash_tables"]
        bits_per_table = config["lsh"]["bits_per_table"]
        hash_bits = config["fingerprint"]["hash_bits"]
        radius= config["lsh"]["radius"]
        lsh_index = BitSamplingLSHIndex(
            radius=radius,hash_bits=hash_bits,num_tables=num_hash_tables, bits_per_table=bits_per_table,seed=42)

    elif lsh_method == "hybrid":
        minhash_params = {
            "num_bands": config["lsh"]["minhash_num_bands"],
            "rows_per_band": config["lsh"]["minhash_rows_per_band"]
        }
        simhash_params = {
            "radius": config["lsh"]["simhash_radius"],
            "hash_bits": config["fingerprint"]["hash_bits"]
        }
        merge_strategy = config["lsh"].get("merge_strategy")
        weights = config["lsh"].get("weights")
        lsh_index = HybridLSHIndex(
            minhash_params, simhash_params, merge_strategy, weights)

    else:
        logger.error(f"未知的 LSH 方法：{lsh_method}")
        return

    lsh_index.index(signatures)
    candidate_pairs = lsh_index.get_candidate_pairs()
    logger.info(f"LSH 索引构建完成，共生成 {len(candidate_pairs)} 个候选对。")
    pipeline_log.add_result("candidate_pairs_count", len(candidate_pairs))
    lsh_index_time = time.time() - lsh_index_time
    pipeline_log.add_runtime("lsh_index_time", lsh_index_time)
    logger.info(f"LSH 索引构建时间：{lsh_index_time:.2f} 秒")

    # 8. 评估
    evaluation_output_path = config["output"]["evaluation_output_path"]
    runtime_log = pipeline_log.runtime_log
    evaluator = Evaluator(candidate_pairs, runtime_log, preprocessed_data)
    duplicate_rate = evaluator.compute_near_duplicate_rate(
        similarity_func=cosine_similarity)
    evaluator.generate_visualizations(output_dir=evaluation_output_path)
    pipeline_log.add_result("duplicate_rate", duplicate_rate)
    logger.info(f"候选对中的近重复文档比率：{duplicate_rate:.2f}")

    # 9. 输出结果
    results_path = config["output"]["results_path"]
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as file:
        for pair in candidate_pairs:
            file.write(f"{pair[0]},{pair[1]}\n")
    logger.info(f"候选对结果已保存至：{results_path}")
    pipeline_log_output_path = config["output"]["pipeline_output_path"]
    pipeline_log.save_log(pipeline_log_output_path)
    logger.info("系统运行完成。")


if __name__ == "__main__":
    main()
