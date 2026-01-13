import pandas as pd
import os
import glob


DATA_DIR = 'Data' 
OUTPUT_DIR = '.'   


COLUMN_MAP = {
    'lambd': 'lambda',
    '3DBall.Environment.CumulativeReward.mean': 'CumulativeReward',
    'Time Elapsed/s': 'TimeElapsed'
}

FEATURES = [
    'batch_size',
    'buffer_size',
    'learning_rate',
    'beta',
    'epsilon',
    'lambda',
    'num_epoch',
    'hidden_units',
    'num_layers',
    'time_horizon',
    'max_steps',
    'gamma',
    'normalize',
    # Hardware (Optional - Tree models can use these)
    'CPU Cores',
    'Total RAM'
]

# TARGETS (The Outputs)
TARGETS = ['TimeElapsed', 'CumulativeReward']


DEFAULTS = {
    'hidden_units': 128,
    'num_layers': 2,
    'gamma': 0.99,
    'max_steps': 500000,
    'time_horizon': 64,
    'batch_size': 64,
    'buffer_size': 10240,
    'learning_rate': 3e-4,
    'beta': 0.001,
    'epsilon': 0.2,
    'lambda': 0.99,
    'num_epoch': 3,
    'normalize': True,   
    'CPU Cores': 8,     
    'Total RAM': 16       # Median guess (GB)
}

def clean_ram(val):
    """Converts '15.14 GB' string to float 15.14"""
    if pd.isna(val): return val
    if isinstance(val, (int, float)): return val
    return float(str(val).replace(' GB', '').replace(' MB', '')) / (1000 if 'MB' in str(val) else 1)

def merge_and_prepare():
    print(" Starting Data Pipeline...")
    
   
    files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    print(f"Found {len(files)} files: {[os.path.basename(f) for f in files]}")
    
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df = df.rename(columns=COLUMN_MAP)
            dfs.append(df)
            print(f"  -> Loaded {os.path.basename(f)} ({len(df)} rows)")
        except Exception as e:
            print(f"   Error reading {f}: {e}")

    if not dfs:
        print(" No data found!")
        return

    full_df = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal Raw Rows: {len(full_df)}")

    
    
    if 'Total RAM' in full_df.columns:
        full_df['Total RAM'] = full_df['Total RAM'].apply(clean_ram)

    
    for col in FEATURES:
        if col not in full_df.columns:
            print(f"   Warning: Column '{col}' missing from all files. Filling with default.")
            full_df[col] = DEFAULTS.get(col, 0)
        
        
        full_df[col] = full_df[col].fillna(DEFAULTS.get(col, 0))


    full_df.dropna(subset=TARGETS, inplace=True)
    
    
    X = full_df[FEATURES]
    y = full_df[TARGETS]

    
    X.to_csv(os.path.join(OUTPUT_DIR, 'X_features.csv'), index=False)
    y.to_csv(os.path.join(OUTPUT_DIR, 'y_targets.csv'), index=False)
    

    combined = pd.concat([X, y], axis=1)
    combined.to_csv(os.path.join(OUTPUT_DIR, 'final_combined_data.csv'), index=False)

    print("\n SUCCESS! Data is ready for RF, XGBoost, and NN.")
    print(f"  -> X_features.csv: {X.shape}")
    print(f"  -> y_targets.csv:  {y.shape}")
    print("  -> final_combined_data.csv (Master)")

if __name__ == "__main__":
    merge_and_prepare()