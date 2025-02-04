import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging
from pathlib import Path
import yaml
import sys

# Configure logging
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

def load_data(path):
    """
    Loads data from a given CSV file path.

    Args:
        path (Path): The file path to load data from.

    Returns:
        pd.DataFrame: Loaded data as a Pandas DataFrame, or None if the file is not found.
    """
    try:
        logging.info(f"Attempting to load data from: {path}")
        if not path.exists():
            logging.error(f"File not found: {path}")
            return None
        data = pd.read_csv(path)
        logging.info(f"Data successfully loaded from: {path}, Shape: {data.shape}")
        return data
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return None

def split_data(data, test_size, random_state):
    """
    Splits the dataset into training and testing sets.

    Args:
        data (pd.DataFrame): The dataset to be split.
        test_size (float): The proportion of the dataset to be included in the test split.
        random_state (int): The seed for random operations.

    Returns:
        tuple: (train_data, test_data)
    """
    try:
        logging.info(f"Splitting data with test_size={test_size} and random_state={random_state}")
        train_data, test_data = train_test_split(data, test_size=test_size, random_state=random_state)
        logging.info(f"Data successfully split. Train shape: {train_data.shape}, Test shape: {test_data.shape}")
        return train_data, test_data
    except Exception as e:
        logging.error(f"Error splitting data: {e}")
        sys.exit(1)  # Exit script if splitting fails

def read_params(file_path):
    """
    Reads configuration parameters from a YAML file.

    Args:
        file_path (Path): The path to the YAML configuration file.

    Returns:
        tuple: (test_size, random_state)
    """
    try:
        logging.info(f"Reading parameters from: {file_path}")
        if not file_path.exists():
            logging.error(f"Parameters file not found: {file_path}")
            sys.exit(1)

        with open(file_path, "r") as f:
            params_file = yaml.safe_load(f)

        parameters = params_file.get('Data_Preparation', {})
        test_size = parameters.get('test_size', 0.2)
        random_state = parameters.get('random_state', 42)
        
        logging.info(f"Parameters loaded: test_size={test_size}, random_state={random_state}")
        return test_size, random_state
    except yaml.YAMLError as e:
        logging.error(f"Error reading YAML file: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Unexpected error reading params: {e}")
        sys.exit(1)

def save_file(path, data, file_name):
    """
    Saves a Pandas DataFrame to a CSV file.

    Args:
        path (Path): The directory where the file should be saved.
        data (pd.DataFrame): The DataFrame to be saved.
        file_name (str): The name of the file (train or test).

    Returns:
        None
    """
    try:
        path.mkdir(exist_ok=True, parents=True)
        file_path = path / f"{file_name}.csv"
        
        data.to_csv(file_path, index=False)
        logging.info(f"Successfully saved {file_name} data to: {file_path}")
    except Exception as e:
        logging.error(f"Error saving {file_name} file: {e}")
        sys.exit(1)

if __name__ == '__main__':
    logging.info("Script execution started")

    try:
        # Define paths
        root_path = Path(__file__).resolve().parent.parent.parent
        file_path = root_path / "data" / "cleaned_ds_ready" / "cleaned_data.csv"
        params_path = root_path / "params.yaml"
        final_path = root_path / "data" / "interim"

        # Load data
        data = load_data(file_path)
        if data is None:
            logging.error("No data found. Exiting script.")
            sys.exit(1)

        # Read parameters
        test_size, random_state = read_params(params_path)

        # Split the data
        train_data, test_data = split_data(data, test_size=test_size, random_state=random_state)

        # Save train and test datasets
        save_file(final_path, train_data, "train")
        save_file(final_path, test_data, "test")

        logging.info("Script execution completed successfully")

    except Exception as e:
        logging.critical(f"Unexpected error in script execution: {e}", exc_info=True)
        sys.exit(1)
