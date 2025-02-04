import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Configure logging
log_file = "data_processing.log"
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
logging.getLogger().addHandler(console_handler)

def read_csv(path):
    """ Reads a CSV file and returns a Data Frame """
    try:
        logging.info(f"Attempting to read file: {path}")
        data = pd.read_csv(path)
        logging.info(f"File successfully loaded: {path}")
        return data
    except FileNotFoundError:
        logging.error(f"File not found at {path}")
        return None
    except Exception as e:
        logging.error(f"Error reading file {path}: {str(e)}")
        return None

def data_cleaning_numerical(data):
    """
    Cleans numerical columns by converting 'price', 'mileage', and 'date' to float.
    Removes the 'Unnamed: 0' column if present.
    """
    try:
        logging.info("Starting numerical data cleaning process.")
        column_clean = ['price', 'mileage']
        if "Unnamed: 0" in data.columns:
            data.drop("Unnamed: 0", axis=1, inplace=True)
            logging.info("Dropped 'Unnamed: 0' column.")
        
        data['date'] = data['date'].replace('na', np.nan).astype(float)
        for col in column_clean:
            data[col] = data[col].replace('na', np.nan)
            data[col] = data[col].str.replace(',', '', regex=True)
            data[col] = data[col].str.replace('$', '', regex=True)
            data[col] = data[col].astype(float)
        
        logging.info("Numerical data cleaning completed successfully.")
        return data
    except Exception as e:
        logging.error(f"Error during numerical data cleaning: {str(e)}")
        return data

def data_clean_categorical(data, categorical_features):
    """
    Cleans categorical data by replacing '0' in 'transmission' with 'Other' and handling missing values.
    """
    try:
        logging.info("Starting categorical data cleaning process.")
        for column in categorical_features:
            if column == 'transmission':
                data[column] = data[column].replace('0', 'Other')
            else:
                data[column] = data[column].replace('na', np.nan)
        
        logging.info("Categorical data cleaning completed successfully.")
        return data
    except Exception as e:
        logging.error(f"Error during categorical data cleaning: {str(e)}")
        return data

def divide_columns(data):
    """
    Splits columns into numerical and categorical features.
    """
    try:
        numerical_features = [feature for feature in data.columns if data[feature].dtype != 'O']
        categorical_features = [feature for feature in data.columns if feature not in numerical_features]
        logging.info(f"Identified {len(numerical_features)} numerical and {len(categorical_features)} categorical features.")
        return numerical_features, categorical_features
    except Exception as e:
        logging.error(f"Error in dividing columns: {str(e)}")
        return [], []

def save_file(path, data):
    """
    Saves cleaned data to a CSV file.
    """
    try:
        path.mkdir(exist_ok=True, parents=True)
        file_path = path / "cleaned_data.csv"
        data.to_csv(file_path, index=False)
        logging.info(f"Cleaned data successfully saved at: {file_path}")
    except Exception as e:
        logging.error(f"Error saving file: {str(e)}")

if __name__ == '__main__':
    try:
        # Finding the root path
        root_path = Path(__file__).parent.parent.parent
        raw_file_path = root_path / "data" / "raw" / "master_data.csv"
        ds_ready_path = root_path / "data" / "cleaned_ds_ready"

        logging.info(f"Script started. Root path: {root_path}")

        # Read Data
        data = read_csv(raw_file_path)
        
        if data is not None:
            logging.info(f"Initial Data Shape: {data.shape}")
            logging.info(f"Data Columns: {list(data.columns)}")
            logging.info(f"Data Info: {data.info()}")

        # Data Cleaning
        data = data_cleaning_numerical(data)
        numerical_features, categorical_features = divide_columns(data)
        data = data_clean_categorical(data, categorical_features)

        # Save Cleaned Data
        save_file(ds_ready_path, data)

        logging.info("Data processing completed successfully.")
    
    except Exception as e:
        logging.critical(f"Critical error occurred: {str(e)}")
