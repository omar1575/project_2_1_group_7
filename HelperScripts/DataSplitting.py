import pandas as pd
import os
from sklearn.model_selection import train_test_split


FOLDER = "Data/"
data_names = ["max.csv", "Moritz_Data.csv", "omar.csv", "robert.csv"]
exclude_columns = ["run_id","timestamp", "DRL Algorithm Used"]

def normal_split(test_size:float):
    df = pd.read_csv(os.path.join(FOLDER, "Data.csv"))
    df = df.drop(columns=exclude_columns)
    return train_test_split(df, test_size=test_size)


# Splitted by each person
def startified_split(test_size:float):
    df_train = pd.DataFrame()
    df_test = pd.DataFrame()
    for filename in data_names:
        df = pd.read_csv(os.path.join(FOLDER, filename))
        df = df.drop(columns=exclude_columns)
        split = train_test_split(df, test_size=test_size)
        df_train = pd.concat([df_train, split[0]])
        df_test = pd.concat([df_test, split[1]])
    return [df_train, df_test]



print(startified_split(0.33))