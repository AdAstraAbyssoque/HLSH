'''
	•	功能：集中配置和管理日志输出。
	•	主要内容：
	•	def setup_logger(log_file: str, level=logging.INFO) -> Logger
	•	设置日志格式、输出到文件和控制台，确保全局调用一致。
'''

import logging
from logging import Logger
from logging.handlers import RotatingFileHandler
import csv


def setup_logger(log_file: str, level=logging.INFO) -> Logger:
    """
    设置日志记录器。
    
    参数:
        log_file (str): 日志文件路径。
        level (int): 日志级别，默认为 logging.INFO。
    
    返回:
        Logger: 配置好的日志记录器。
    """
    # 创建日志记录器
    
    logger = logging.getLogger(log_file)
    logger.setLevel(level)
    
    # 防止重复添加处理器
    if not logger.handlers:
        # 创建日志格式
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # 文件处理器（带日志轮转功能）
        file_handler = RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024, backupCount=3)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger

class Log_pipeline_info:
    """
    记录整个pipeling中的参数和结果
    参数：
        runtime_log(dict[str,float])：运行时间
        params(dict)：参数
        results(dict)：结果
    """
    def __init__(self,params=None):
        self.runtime_log = {}
        self.params = params if params else {}
        self.results = {}
    
    def add_runtime(self, stage: str, time: float):
        """
        添加运行时间信息。
        
        参数:
            stage (str): 阶段名称。
            time (float): 运行时间（秒）。
        """
        self.runtime_log[stage] = time
    def add_param(self, param_name: str, param_value):
        """
        添加参数信息。
        """
        self.params[param_name] = param_value
        
    def add_result(self, result_name: str, result_value):
        """
        添加结果信息。
        
        参数:
            result_name (str): 结果名称。
            result_value: 结果值。
        """
        self.results[result_name] = result_value


    def save_log(self, log_file: str):
        """
        保存日志信息到 CSV 文件。

        参数:
            log_file (str): 日志文件路径。
        """
        with open(log_file, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)

            # 写入运行时间
            writer.writerow(["运行时间"])
            writer.writerow(["阶段", "时间 (秒)"])
            for stage, time in self.runtime_log.items():
                writer.writerow([stage, time])

            # 写入参数
            writer.writerow([])  # 空行分隔
            writer.writerow(["参数"])
            writer.writerow(["参数名称", "参数值"])
            for param_name, param_value in self.params.items():
                writer.writerow([param_name, param_value])

            # 写入结果
            writer.writerow([])  # 空行分隔
            writer.writerow(["结果"])
            writer.writerow(["结果名称", "结果值"])
            for result_name, result_value in self.results.items():
                writer.writerow([result_name, result_value])


# 示例用法
if __name__ == "__main__":
    logger = setup_logger("app.log")
    logger.info("日志记录器已成功配置！")
    logger.warning("这是一个警告日志示例。")
    logger.error("这是一个错误日志示例。")