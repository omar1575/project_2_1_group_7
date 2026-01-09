import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from catboost import CatBoostRegressor

data_file = "Data/Data.csv"
data = pd.read_csv(data_file).drop(columns=['run_id', 'timestamp'], errors='ignore')
print("Data loaded")

mask = ~data.isna().any(axis=1)
data = data[mask]


int_columns = ["cpu_frequency", "total_ram"]
for col in int_columns:
    data[col] = pd.to_numeric(data[col].astype(str).str.replace(r'[a-zA-Z\s]', '', regex=True), errors='coerce')


cat_cols = [] 


for col in data.columns: 
    if data[col].dtype == 'object' and data[col].nunique() < 20:  
        cat_cols.append(col) 



target_cols = ['time_elapased_seconds', 'cumulative_reward_mean']
X = data.drop(columns=target_cols)
y = data[target_cols]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


y1 = y_train['time_elapased_seconds']
y2 = y_train['cumulative_reward_mean']


train_preds = [None, None]

for i, y_col_train in enumerate([y1, y2]):
    model = CatBoostRegressor(loss_function='RMSE')
    model.fit(X_train, y_col_train, verbose=100, cat_features=cat_cols)
    train_preds[i] = model.predict(X_test)
    
preds = np.stack(train_preds, axis=1)
mae = mean_absolute_error(y_test, preds, multioutput='raw_values')

y_test_np = y_test.to_numpy()

print("FULL TEST SET PREDICTIONS (Time, Perf):")
for i in range(len(y_test)):
    print(f"Row {i+1}: Pred {np.round(preds[i], 2)} | Actual {np.round(y_test_np[i], 2)}")

print(f"OVERALL ERROR -> Time: {mae[0]:.2f}s | Perf: {mae[1]:.2f} pts\n")


