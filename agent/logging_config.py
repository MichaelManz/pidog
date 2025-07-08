"""
Centralized logging configuration for LangGraph debugging.

This module provides consistent logging setup across all files and makes it easy
to exclude noisy loggers in one place.
"""

import logging
import os
import re

def truncate_at_base64(text, max_length=500):
    """Truncate text and replace base64 content with placeholder"""
    if "base64," in text:
        # Find and replace base64 content with placeholder
        pattern = r'base64,[A-Za-z0-9+/=]+'
        text = re.sub(pattern, 'base64,<...BASE64_CONTENT_TRUNCATED...>', text)
    
    if len(text) > max_length:
        return text[:max_length] + "..."
    return text

def setup_logging(log_level=logging.DEBUG, log_file=None, console_output=True):
    """
    Set up logging configuration with noise reduction.
    
    Args:
        log_level: Base logging level (default: DEBUG)
        log_file: Optional log file path
        console_output: Whether to output to console (default: True)
    """
    
    # Create handlers
    handlers = []
    
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        handlers.append(file_handler)
    
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        handlers.append(console_handler)
    
    # Configure basic logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers,
        force=True  # Override existing configuration
    )
    
    # Set specific loggers to higher levels to reduce noise
    noisy_loggers = [
        'picamera2.picamera2',
        'picamera2',
        'libcamera',
        'PIL',
        'urllib3',
        'requests',
        'vilib',
        'httpx',
        'httpcore',
        'matplotlib',
        'numpy',
        'openai._base_client'
    ]
    
    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    
    # You can add more specific configurations here
    # For example, to completely silence a logger:
    # logging.getLogger('some_very_noisy_logger').setLevel(logging.CRITICAL)

def get_logger(name):
    """
    Get a logger with the standard configuration.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)

def exclude_logger(logger_name, level=logging.WARNING):
    """
    Exclude a specific logger by setting its level.
    
    Args:
        logger_name: Name of the logger to exclude
        level: Log level to set (default: WARNING)
    """
    logging.getLogger(logger_name).setLevel(level)

def silence_logger(logger_name):
    """
    Completely silence a logger.
    
    Args:
        logger_name: Name of the logger to silence
    """
    logging.getLogger(logger_name).setLevel(logging.CRITICAL + 1)

# Example usage patterns:
# 
# # Basic setup
# from logging_config import setup_logging, get_logger
# setup_logging(log_file='debug.log')
# logger = get_logger(__name__)
# 
# # Exclude additional loggers
# from logging_config import exclude_logger
# exclude_logger('my_noisy_library')
# 
# # Completely silence a logger
# from logging_config import silence_logger
# silence_logger('extremely_verbose_library') 