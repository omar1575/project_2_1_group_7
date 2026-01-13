import pandas as pd
import numpy as np
import preprocess
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

from catboost import CatBoostRegressor
from MLModel import Model



class CatBoostModel(Model):
    def __init__(self):
        self.models_ = None
    
    def fit(self, X:pd.DataFrame, y:pd.DataFrame) -> None:
        
        # Get categorical columns
        cat_cols = [] 

        for col in X.columns: 
            if X[col].dtype == 'object':  
                cat_cols.append(col) 

        #Divide target columns
        y_cols = [y['time_elapased_seconds'], y['cumulative_reward_mean']]


        self.models_ = [None, None]
        #Train new model on each target seperately
        for i, y_col_train in enumerate(y_cols):
            model = CatBoostRegressor(loss_function='RMSE')
            model.fit(X, y_col_train, verbose=100, cat_features=cat_cols)
            self.models_[i] = model


    def predict(self, X:pd.DataFrame) -> np.ndarray:
        if self.models_ is None:
           raise RuntimeError("Model not fitted yet")
       
        col_preds = [None, None]
       
        for i, model in enumerate(self.models_):
            col_preds[i] = model.predict(X)
        preds = np.stack(col_preds, axis=1)

        return preds
    


if __name__ == "__main__":
    
    data = preprocess.get_processed_data()
    # Divide data in split and non-split

    target_cols = ['time_elapased_seconds', 'cumulative_reward_mean']
    X = data.drop(columns=target_cols)
    y = data[target_cols]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = CatBoostModel()

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    model.save("CatBoost")

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
