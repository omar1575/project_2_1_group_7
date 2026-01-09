import pandas as pd
import numpy as np
import preprocess
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

from catboost import CatBoostRegressor


data = preprocess.get_processed_data()

# Get categorical columns
cat_cols = [] 

for col in data.columns: 
    if data[col].dtype == 'object':  
        cat_cols.append(col) 


# Divide data in split and non-split
target_cols = ['time_elapased_seconds', 'cumulative_reward_mean']
X = data.drop(columns=target_cols)
y = data[target_cols]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Divide target columns
y_train_time = y_train['time_elapased_seconds']
y_train_time = y_train['cumulative_reward_mean']


train_preds = [None, None]

#Train new model on each target seperately
for i, y_col_train in enumerate([y_train_time, y_train_time]):
    model = CatBoostRegressor(loss_function='RMSE')
    model.fit(X_train, y_col_train, verbose=100, cat_features=cat_cols)
    train_preds[i] = model.predict(X_test)
    
#Construct predictions as from multitarget predict
preds = np.stack(train_preds, axis=1)

mse = mean_squared_error(y_test, preds, multioutput='raw_values')
mae = mean_absolute_error(y_test, preds, multioutput='raw_values')

#print each row in testing
y_test_np = y_test.to_numpy()
print("FULL TEST SET PREDICTIONS (Time, Perf):")
for i in range(len(y_test)):
    print(f"Row {i+1}: Pred {np.round(preds[i], 2)} | Actual {np.round(y_test_np[i], 2)}")

#print overall errors
print(f"OVERALL MSE ERROR -> Time: {mse[0]:.2f}s | Perf: {mse[1]:.2f} pts\n")
print(f"OVERALL MAE ERROR -> Time: {mae[0]:.2f}s | Perf: {mae[1]:.2f} pts\n")


