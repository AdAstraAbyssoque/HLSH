'''
	•	功能：集中配置和管理日志输出。
	•	主要内容：
	•	def setup_logger(log_file: str, level=logging.INFO) -> Logger
	•	设置日志格式、输出到文件和控制台，确保全局调用一致。
'''

import logging
from logging import Logger
from logging.handlers import RotatingFileHandler

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

# 示例用法
if __name__ == "__main__":
    logger = setup_logger("app.log")
    logger.info("日志记录器已成功配置！")
    logger.warning("这是一个警告日志示例。")
    logger.error("这是一个错误日志示例。")