import pandas as pd
import os
from src.constants import RAW_DATA_DIR, DATA_FILENAME

class DataLoader:
    def __init__(self, data_dir=RAW_DATA_DIR, data_file=DATA_FILENAME):
        self.data_path = os.path.join(data_dir, data_file)

    def load_data(self):
        """Load data from the specified CSV file."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found at {self.data_path}")
        
        data = pd.read_csv(self.data_path)
        return data

