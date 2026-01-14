import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from MLModel import Model

class RandomForestModel(Model):
    def __init__(self, n_estimators: int = 100, random_state: int = 42, n_jobs: int = -1, max_depth: int = None):
        super().__init__()
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.max_depth = max_depth
        self.models_ = None

    def fit(self, X:pd.DataFrame, y:pd.DataFrame) -> None:
        X_encoded = pd.get_dummies(X, drop_first=True)

        target_cols = ['time_elapased_seconds', 'cumulative_reward_mean']
        y_time = y[target_cols[0]]
        y_reward = y[target_cols[1]]

        self.models_ = []

        rf_time = RandomForestRegressor(n_estimators=self.n_estimators, random_state=self.random_state, n_jobs=self.n_jobs, max_depth=self.max_depth)
        rf_time.fit(X_encoded, y_time)
        self.models_.append(rf_time)
        
        rf_reward = RandomForestRegressor(n_estimators=self.n_estimators, random_state=self.random_state, n_jobs=self.n_jobs, max_depth=self.max_depth)
        rf_reward.fit(X_encoded, y_reward)
        self.models_.append(rf_reward)

        self.feature_names_ = X_encoded.columns.tolist()

    def predict(self, X:pd.DataFrame) -> np.ndarray:
        if self.models_ is None:
           raise RuntimeError("Model not fitted yet. Call fit() before predict().")
        
        X_encoded = pd.get_dummies(X, drop_first=True)
    
        for col in self.feature_names_:
            if col not in X_encoded.columns:
                X_encoded[col] = 0
        X_encoded = X_encoded[self.feature_names_]

        predictions = []
        for model in self.models_:
            predictions.append(model.predict(X_encoded))

        return np.stack(predictions, axis=1)
