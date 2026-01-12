# utlility module containing common functions and helpers for the entire project


# logging functions for the project
import logging
import os

def setup_logger(log_dir='logs', log_file = "app.log"):
    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, log_file)),
            logging.StreamHandler()
        ]
    )