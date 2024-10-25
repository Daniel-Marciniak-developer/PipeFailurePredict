import pandas as pd

class DataLoader:
    """
    A utility class for loading data from CSV files into pandas DataFrames.
    """

    @staticmethod
    def load_demands(file_path: str) -> pd.DataFrame:
        """
        Loads demand data from a CSV file.
        
        Args:
            file_path (str): The path to the demand CSV file.
        
        Returns:
            pd.DataFrame: The demand data.
        """
        return pd.read_csv(file_path)

    @staticmethod
    def load_pipes(file_path: str) -> pd.DataFrame:
        """
        Loads pipe data from a CSV file.
        
        Args:
            file_path (str): The path to the pipe network CSV file.
        
        Returns:
            pd.DataFrame: The pipe network data.
        """
        return pd.read_csv(file_path)
