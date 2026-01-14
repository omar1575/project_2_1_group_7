import pandas as pd

def get_processed_data(data_file: str = "Data/Data.csv") -> pd.DataFrame:
    data = pd.read_csv(data_file).drop(columns=['run_id', 'timestamp'], errors='ignore')

    data = data.dropna()

    int_columns = ["cpu_frequency", "total_ram"]
    for col in int_columns:
        data[col] = pd.to_numeric(data[col].astype(str).str.replace(r'[a-zA-Z\s]', '', regex=True), errors='coerce')

    return data

def split_features_targets(data: pd.DataFrame, target_cols: list = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    if target_cols is None:
        target_cols = ['time_elapased_seconds', 'cumulative_reward_mean']

    X = data.drop(columns=target_cols, errors='ignore')
    y = data[target_cols]
    
    return X, y
