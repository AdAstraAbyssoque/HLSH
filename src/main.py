'''
    • Functionality: Serves as the entry point for the entire system, responsible for orchestrating the workflow.
    • Main content:
    • Configuration import: Loads config/config.yaml.
    • Logging initialization: Calls the setup function from utils/logger.py.
    • Data loading: Uses utils/data_loader.py to read raw data.
    • Preprocessing: Calls classes or functions from preprocessing.py to clean, tokenize, and generate n-grams from raw data.
    • Feature extraction: Calls feature_extraction.py to generate text feature representations (e.g., sets or vectors).
    • Fingerprint generation: Based on the configuration, calls classes and methods from fingerprint/minhash.py, fingerprint/simhash.py, and fingerprint/bitsampling.py to generate text fingerprints.
    • LSH index construction: Passes various signatures to lsh/lsh_index.py to build an LSH index and filter candidate document pairs.
    • Evaluation: Calls lsh/evaluation.py or evaluation.py to quantitatively and qualitatively evaluate the generated candidate pairs.
    • Output results: Writes evaluation reports, logs, and candidate pair results to the data/results/ directory.
    • Invocation relationship: Acts as the top-level entry point for other modules, forming a complete pipeline by sequentially invoking preprocessing, feature extraction, fingerprint generation, LSH indexing, and evaluation modules.
'''
import os
import yaml
import time
import gc
import pandas as pd
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
from joblib import Parallel, delayed
from preprocessing import Preprocessor
from lsh.helper import cosine_similarity, jaccard_similarity, euclidean_distance
from preprocessing import preprocess_text, parallel_preprocess_texts


# Load configuration file
def load_config(config_path: str) -> dict:
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def main():
    # 1. Load configuration
    config_path = "config/config.yaml"
    config = load_config(config_path)

    # 1.1 Get parallel configuration
    parallel_config = config.get("parallel", {})
    parallel_enabled = parallel_config.get("enabled", False)
    thread_pool_size = parallel_config.get("thread_pool_size", 4)
    process_pool_size = parallel_config.get("process_pool_size", 8)

    # 2. Initialize logging and time tracking
    log_file = config["logging"]["log_file"]
    log_level = config["logging"]["log_level"]
    logger = setup_logger(log_file, log_level)
    logger.info("-------------------------------------------------------")
    logger.info("System started, configuration loaded.")

    # Initialize pipeline log parameters
    pipeline_log = Log_pipeline_info(config)

    # 3. Data loading
    data_loader_time = time.time()
    data_loader = DataLoader()
    raw_data_path = config["data"]["raw_data_path"]  # Can be a file path or directory path
    logger.info(f"Loading data from path: {raw_data_path}")

    raw_data = []
    data_loader_parallel = parallel_config.get("data_loader_parallel", False)
    raw_data = data_loader.load_data(
        raw_data_path, parallel_enabled=data_loader_parallel, thread_pool_size=thread_pool_size)

    logger.info(f"Data loading completed, {len(raw_data)} records loaded.")
    pipeline_log.add_result("raw_data_count", len(raw_data))
    data_loader_time = time.time() - data_loader_time
    pipeline_log.add_runtime("data_loader_time", data_loader_time)
    logger.info(f"Data loading time: {data_loader_time:.2f} seconds")

    # raw_data = raw_data[:4000]  # For testing, only take the first 1000 records

    # 4. Data preprocessing
    preprocess_data_time = time.time()
    logger.info("Starting data preprocessing...")

    preprocess_parallel = parallel_config.get("preprocess_parallel", True)
    preprocessed_data = parallel_preprocess_texts(
        raw_data, process_pool_size=process_pool_size, parallel_enabled=preprocess_parallel
    )

    logger.info("Data preprocessing completed.")
    
    # Save preprocessed data as CSV
    preprocessed_csv_path = os.path.join("data", "processed", "preprocessed_data.csv")
    os.makedirs(os.path.dirname(preprocessed_csv_path), exist_ok=True)
    pd.DataFrame({"text": preprocessed_data}).to_csv(preprocessed_csv_path, index=False)
    logger.info(f"Preprocessed data saved as CSV file: {preprocessed_csv_path}")
    
    preprocess_data_time = time.time() - preprocess_data_time
    pipeline_log.add_runtime("preprocess_data_time", preprocess_data_time)
    logger.info(f"Data preprocessing time: {preprocess_data_time:.2f} seconds")

    # 5. Feature extraction
    feature_extraction_time = time.time()
    feature_method = config["feature_extraction"]["method"]
    ngram_size = config["feature_extraction"].get("ngram_size", 3)

    logger.info(f"Starting feature extraction, method: {feature_method}")
    extractor = FeatureExtractor(method=feature_method, n=ngram_size)
    feature_extraction_parallel = parallel_config.get(
        "feature_extraction_parallel", True)
    logger.info("Parallel enabled" if feature_extraction_parallel else "Parallel disabled")

    # Call the parallel feature extraction method of FeatureExtractor
    features = extractor.parallel_extract_features(
        preprocessed_data, process_pool_size=process_pool_size, parallel_enabled=feature_extraction_parallel
    )

    idf = None
    if feature_method == "vectorize":
        token_for_simhash = [extractor._extract_ngrams(
            text) for text in preprocessed_data]
        idf, features = features, token_for_simhash

    logger.info("Feature extraction completed.")
    feature_extraction_time = time.time() - feature_extraction_time
    pipeline_log.add_runtime("feature_extraction_time",
                             feature_extraction_time)
    logger.info(f"Feature extraction time: {feature_extraction_time:.2f} seconds")


    # 6. Fingerprint generation
    fingerprint_time = time.time()
    fingerprint_method = config["fingerprint"]["method"]
    use_cache = parallel_config.get("use_memory_cache", True)
    logger.info(f"Starting fingerprint generation, method: {fingerprint_method}")
    fingerprint_parallel = parallel_config.get("fingerprint_parallel", True)
    logger.info("Parallel enabled" if fingerprint_parallel else "Parallel disabled")

    if fingerprint_method == "hybrid":
        # Hybrid method requires generating both MinHash and SimHash signatures
        num_hashes = config["fingerprint"]["num_hashes"]
        hash_bits = config["fingerprint"]["hash_bits"]
        seed = config["fingerprint"].get("seed", None)

        minhash = MinHash(num_hashes=num_hashes, seed=seed)
        simhash = SimHash(hash_bits=hash_bits)

        # Generate MinHash signatures
        minhash_signatures = minhash.parallel_compute_signature(
            features, parallel_enable=fingerprint_parallel, process_pool_size=process_pool_size
        )
        # Generate SimHash signatures
        simhash_signatures = simhash.parallel_compute_signature(
            features, idf=idf, parallel_enable=fingerprint_parallel, process_pool_size=process_pool_size
        )
        # Combine signatures
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

    # If memory cache is enabled, clean up the process pool
    if use_cache:
        gc.collect()

    logger.info("Fingerprint generation completed, total signatures generated: {}".format(len(signatures)))
    pipeline_log.add_result("signature_count", len(signatures))
    fingerprint_time = time.time() - fingerprint_time
    pipeline_log.add_runtime("fingerprint_time", fingerprint_time)
    logger.info(f"Fingerprint generation time: {fingerprint_time:.2f} seconds")

    # Use DataLoader to save fingerprint signatures
    fingerprint_output_path = config["output"]["fingerpritnts_path"]
    try:
        logger.info(f"Saving fingerprint signatures to file: {fingerprint_output_path}")
        data_loader.save_signatures(signatures, fingerprint_output_path)
        logger.info(f"Fingerprint signatures successfully saved to file: {fingerprint_output_path}")
    except Exception as e:
        logger.error(f"Failed to save fingerprint signatures, error: {e}")
        return
    
    # 7. LSH index construction
    lsh_index_time = time.time()
    lsh_method = config["lsh"]["method"]
    logger.info(f"Starting LSH index construction, method: {lsh_method}")

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
        logger.error(f"Unknown LSH method: {lsh_method}")
        return

    lsh_index.index(signatures)
    candidate_pairs = lsh_index.get_candidate_pairs()
    logger.info(f"LSH index construction completed, {len(candidate_pairs)} candidate pairs generated.")
    pipeline_log.add_result("candidate_pairs_count", len(candidate_pairs))
    lsh_index_time = time.time() - lsh_index_time
    pipeline_log.add_runtime("lsh_index_time", lsh_index_time)
    logger.info(f"LSH index construction time: {lsh_index_time:.2f} seconds")

    # 8. Evaluation
    evaluation_output_path = config["output"]["evaluation_output_path"]
    runtime_log = pipeline_log.runtime_log
    evaluator = Evaluator(candidate_pairs, runtime_log, preprocessed_data)
    duplicate_rate = evaluator.compute_near_duplicate_rate(
        similarity_func=cosine_similarity)
    evaluator.generate_visualizations(output_dir=evaluation_output_path)
    pipeline_log.add_result("duplicate_rate", duplicate_rate)
    logger.info(f"Near-duplicate document rate in candidate pairs: {duplicate_rate:.2f}")

    # 9. Output results
    results_path = config["output"]["results_path"]
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as file:
        for pair in candidate_pairs:
            file.write(f"{pair[0]},{pair[1]}\n")
    logger.info(f"Candidate pair results saved to: {results_path}")
    pipeline_log_output_path = config["output"]["pipeline_output_path"]
    pipeline_log.save_log(pipeline_log_output_path)
    logger.info("System execution completed.")


if __name__ == "__main__":
    main()