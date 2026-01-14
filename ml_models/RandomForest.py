import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

csv_path = "Data/final_combined_data_L.csv"
df = pd.read_csv(csv_path)
df = df.sample(frac=1.0, random_state=17).reset_index(drop=True)

features = [
    "batch_size",
    "buffer_size",
    "learning_rate",
    "beta",
    "epsilon",
    "lambda",
    "num_epoch",
    "hidden_units",
    "num_layers",
    "gamma",
    "time_horizon",
]

target_time = "TimeElapsed"
target_reward = "CumulativeReward"

df = df.dropna()

X = df[features]
Y_time = df[target_time]
Y_reward = df[target_reward]

best_time_MSE = 1000000
best_reward_MSE = 1000000

bestRandomState = -1
bestEstimatorsNum = -1

for estimators in range(1,500):
    print("Estimators: ", estimators)
    
    X_train, X_test, Y_time_train, Y_time_test, Y_reward_train, Y_reward_test = train_test_split(X, Y_time, Y_reward, test_size=0.2, random_state=17)

    rf_time = RandomForestRegressor(n_estimators=estimators, random_state=17, n_jobs=-1)
    rf_time.fit(X_train, Y_time_train)

    rf_reward = RandomForestRegressor(n_estimators=estimators, random_state=17, n_jobs=-1)
    rf_reward.fit(X_train, Y_reward_train)

    rf_time_predictions = rf_time.predict(X_test)
    rf_reward_predictions = rf_reward.predict(X_test)

    if mean_squared_error(Y_time_test, rf_time_predictions) < best_time_MSE and mean_squared_error(Y_reward_test, rf_reward_predictions) < best_reward_MSE:
        bestEstimatorsNum = estimators

        
print(bestEstimatorsNum)