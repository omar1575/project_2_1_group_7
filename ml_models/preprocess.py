import pandas as pd


data_file = "Data/Data.csv"

def get_processed_data() -> pd.DataFrame:
    
    #Get data and remove unneded columns(ids and time)
    data = pd.read_csv(data_file).drop(columns=['run_id', 'timestamp'], errors='ignore')

    #Remove rows containing empty rows
    mask = ~data.isna().any(axis=1)
    data = data[mask]

    # rewrite string rows to numeric values 
    int_columns = ["cpu_frequency", "total_ram"]
    for col in int_columns:
        data[col] = pd.to_numeric(data[col].astype(str).str.replace(r'[a-zA-Z\s]', '', regex=True), errors='coerce')

    return data