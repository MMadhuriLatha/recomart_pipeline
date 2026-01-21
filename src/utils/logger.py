import logging
import os
from datetime import datetime
from pathlib import Path

def setup_logger(name, log_dir='logs', level=logging.INFO):
    """
    Setup logger with file and console handlers
    
    Args:
        name: Logger name
        log_dir: Directory to store log files
        level: Logging level
    
    Returns:
        Logger instance
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'{name}_{timestamp}.log')
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if logger.handlers:
        logger.handlers.clear()
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

class PipelineLogger:
    """Context manager for pipeline logging"""
    
    def __init__(self, stage_name, log_dir='logs'):
        self.stage_name = stage_name
        self.logger = setup_logger(stage_name, log_dir)
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.info(f"Starting {self.stage_name}")
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (datetime.now() - self.start_time).total_seconds()
        
        if exc_type is None:
            self.logger.info(f"Completed {self.stage_name} in {duration:.2f}s")
        else:
            self.logger.error(f"Failed {self.stage_name} after {duration:.2f}s")
            self.logger.error(f"Error: {exc_val}")
        
        return False