from __future__ import annotations
import numpy as np
import pandas as pd
import pickle
from abc import ABC, abstractmethod

class Model(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def fit(self, X:pd.DataFrame, y:pd.DataFrame) -> None:
        pass

    @abstractmethod
    def predict(self, X:pd.DataFrame) -> np.ndarray:
        pass

    def save(self, name: str) -> None:
        pickle.dump(self, open(name, 'wb'))

    @staticmethod
    def load(name) -> Model:
        loaded_model = pickle.load(open(name, 'rb'))
        return loaded_model
    
    def get_name(self) -> str:
        return self.__class__.__name__
