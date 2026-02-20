import pandas as pd

def load_data(path):
    """
    Load dataset from given path
    """
    data = pd.read_csv(path)
    return data


def show_basic_info(data):
    """
    Display dataset basic information
    """
    print("\nFirst 5 Rows:\n", data.head())
    print("\nShape:", data.shape)
    print("\nMissing Values:\n", data.isnull().sum())
    print("\nSummary Statistics:\n", data.describe())
