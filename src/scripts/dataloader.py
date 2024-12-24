import pandas as pd
import os

class DataLoader:
    """
    Class responsible for loading data from a URL or local file.
    """
    def __init__(self, path: str):
        """
        :param path: URL or path to the CSV file.
        """
        self.path = path

    def load_data(self) -> pd.DataFrame:
        """
        Loads the dataset from a URL or local path.
        :return: Pandas DataFrame with the data.
        """
        if self.path.startswith("http"):
            #----- Load from URL
            print(f"Loading data from the URL: {self.path}")
            data = pd.read_csv(self.path)
        elif os.path.exists(self.path):
            #----- Load from local file
            print(f"Loading data from local file: {self.path}")
            data = pd.read_csv(self.path)
        else:
            raise FileNotFoundError(f"File not found at the provided path: {self.path}")
        return data
