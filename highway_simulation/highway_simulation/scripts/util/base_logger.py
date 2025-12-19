"""Logger setup for the simulation."""

import logging

def setup_logger(
    name: str = "main_logger",
    log_file: str = "ego_vehicle.log",
    level: int = logging.DEBUG,
) -> logging.Logger:
    """Sets up a logger with specified name and configuration."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid adding handlers multiple times
    if not logger.handlers:
        # Create a file handler
        file_handler = logging.FileHandler(log_file, mode="w")
        file_handler.setLevel(level)
        
        # Create a console handler (optional)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Define the formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger

# Export a global logger instance
logger = setup_logger()
