'''
MLP Regressor to predict training time and performance from hyperparameters and system specs. MLP Regressor is a feed-forward neural network (multi-layer perceptron) suitable for regression tasks.
'''
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error

data_file = "Data/Data.csv"

# Load and preprocess data
data = pd.read_csv(data_file).drop(columns=['run_id', 'timestamp'], errors='ignore')

# Convert object columns with numeric strings to numeric types
for col in data.columns:
    if data[col].dtype == 'object' and data[col].astype(str).str.contains(r'\d').any():
        data[col] = pd.to_numeric(data[col].astype(str).str.replace(r'[a-zA-Z\s]', '', regex=True), errors='coerce')

# One-hot encode categorical variables and handle missing values
data_encoded = pd.get_dummies(data)
data_encoded = data_encoded.loc[:, data_encoded.nunique() > 1].fillna(0)

# Split data into features and targets
target_cols = ['time_elapased_seconds', 'cumulative_reward_mean']
X = data_encoded.drop(columns=target_cols).values
y = data_encoded[target_cols].values

# 80% train, 20% test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling features and targets to feed normalized data to the MLP
sc_X, sc_y = StandardScaler(), StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
y_train_s = sc_y.fit_transform(y_train)

# Define and train MLP Regressor model
model = MLPRegressor(hidden_layer_sizes=(256, 128, 64), max_iter=5000, early_stopping=True, random_state=42)
model.fit(X_train, y_train_s)

# Prediction and inverse scaling
preds = sc_y.inverse_transform(model.predict(X_test))

# Evaluate model performance using Mean Absolute Error
mae = mean_absolute_error(y_test, preds, multioutput='raw_values')

print("FULL TEST SET PREDICTIONS (Time, Perf):")
for i in range(len(y_test)):
    print(f"Row {i+1}: Pred {np.round(preds[i], 2)} | Actual {np.round(y_test[i], 2)}")

print(f"OVERALL ERROR -> Time: {mae[0]:.2f}s | Perf: {mae[1]:.2f} pts\n")