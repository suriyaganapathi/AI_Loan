import os
import logging
from logging.handlers import RotatingFileHandler
from fastapi import UploadFile
import pandas as pd

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Application configuration"""
    MAX_FILE_SIZE_MB = 50
    ALLOWED_EXTENSIONS = ['.xlsx', '.xls', '.csv']
    CHUNK_SIZE = 5000  # Process large files in chunks
    LOG_DIR = 'logs'
    LOG_FILE = 'api.log'
    LOG_LEVEL = logging.INFO


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging():
    """Configure comprehensive logging"""
    # Create logs directory if it doesn't exist
    os.makedirs(Config.LOG_DIR, exist_ok=True)
    
    # Configure logger
    logger = logging.getLogger('categorization_api')
    logger.setLevel(Config.LOG_LEVEL)
    
    # Rotating file handler (max 10MB, keep 5 backups)
    file_handler = RotatingFileHandler(
        os.path.join(Config.LOG_DIR, Config.LOG_FILE),
        maxBytes=10*1024*1024,
        backupCount=5
    )
    file_handler.setLevel(Config.LOG_LEVEL)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(Config.LOG_LEVEL)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Initialize logger for the module
logger = setup_logging()


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def validate_file_size(file: UploadFile) -> bool:
    """Validate uploaded file size"""
    # Read file to check size
    file.file.seek(0, 2)  # Seek to end
    file_size = file.file.tell()  # Get position (size)
    file.file.seek(0)  # Reset to start
    
    size_mb = file_size / (1024 * 1024)
    if size_mb > Config.MAX_FILE_SIZE_MB:
        logger.warning(f"File too large: {size_mb:.2f}MB (max: {Config.MAX_FILE_SIZE_MB}MB)")
        return False
    
    logger.info(f"File size: {size_mb:.2f}MB")
    return True


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to handle variations"""
    # Remove newlines and extra spaces
    df.columns = [str(col).replace('\n', ' ').strip() for col in df.columns]
    return df


def optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Optimize dataframe for memory efficiency"""
    # Convert STATUS to category type for better performance
    if 'STATUS' in df.columns:
        df['STATUS'] = df['STATUS'].astype('category')
    return df
