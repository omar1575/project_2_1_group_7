import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from MLModel import Model

class CatBoostModel(Model):
    def __init__(self, iterations: int = 1000, learning_rate: float = 0.1, depth: int = 6, verbose: bool = False):
        super().__init__()
        self.models_ = None
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.verbose = verbose
        self.cat_features = None

    def fit(self, X:pd.DataFrame, y:pd.DataFrame) -> None:
        # Identify categorical columns
        self.cat_features = [col for col in X.columns if X[col].dtype == 'object']

        #Divide target columns
        target_cols = ['time_elapased_seconds', 'cumulative_reward_mean']
        y_cols = [y[target_cols[0]], y[target_cols[1]]]

        # Train a separate CatBoost model for each target
        self.models_ = []
        for i, y_col in enumerate(y_cols):
            model = CatBoostRegressor(
                iterations=self.iterations,
                learning_rate=self.learning_rate,
                depth=self.depth,
                loss_function='RMSE',
                allow_writing_files=False,
                verbose=self.verbose
            )
            model.fit(X, y_col, cat_features=self.cat_features)
            self.models_.append(model)

    def predict(self, X:pd.DataFrame) -> np.ndarray:
        if self.models_ is None:
           raise RuntimeError("Model not fitted yet. Call fit() before predict().")
       
        predictions = []
        for model in self.models_:
            predictions.append(model.predict(X))

        return np.stack(predictions, axis=1)
    