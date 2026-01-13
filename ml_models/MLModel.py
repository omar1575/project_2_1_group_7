import numpy as np
import pandas as pd
import pickle
from __future__ import annotations



class Model():
    def __init__(self):
        pass
    
    def fit(self, df:pd.DataFrame) -> None:
        pass

    def predict(self, df:pd.DataFrame) -> np.ndarray:
        pass

    def save(self, name: str) -> None:
        pickle.dump(self, open(name, 'wb'))

    @staticmethod
    def load(name) -> Model:
        loaded_model = pickle.load(open(name, 'rb'))
        return loaded_model
    

