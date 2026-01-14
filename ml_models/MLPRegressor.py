'''
MLP Regressor to predict training time and performance from hyperparameters and system specs. MLP Regressor is a feed-forward neural network (multi-layer perceptron) suitable for regression tasks.
'''
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from MLModel import Model

class MLPModel(Model):
    def __init__(self, hidden_layer_sizes: tuple, max_iter: int = 5000, early_stopping: bool = True, random_state: int = 42):
        super().__init__()
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        self.random_state = random_state

        self.model_ = None
        self.scaler_X = None
        self.scaler_y = None

    def fit(self, X:pd.DataFrame, y:pd.DataFrame) -> None:
        X_encoded = pd.get_dummies(X, drop_first=True)
        X_encoded = X_encoded.loc[:, X_encoded.nunique() > 1]

        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()

        X_scaled = self.scaler_X.fit_transform(X_encoded)
        y_scaled = self.scaler_y.fit_transform(y)

        self.model_ = MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            max_iter=self.max_iter,
            early_stopping=self.early_stopping,
            random_state=self.random_state
        )
        self.model_.fit(X_scaled, y_scaled)

        self.feature_columns_ = X_encoded.columns.tolist()

    def predict(self, X:pd.DataFrame) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Model not fitted yet. Call fit() before predict().")

        X_encoded = pd.get_dummies(X, drop_first=True)

        for col in self.feature_columns_:
            if col not in X_encoded.columns:
                X_encoded[col] = 0
        X_encoded = X_encoded[self.feature_columns_]

        X_scaled = self.scaler_X.transform(X_encoded)

        y_pred_scaled = self.model_.predict(X_scaled)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)

        X_encoded = X_encoded.reindex(columns=self.feature_columns_, fill_value=0)

        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
        return y_pred
