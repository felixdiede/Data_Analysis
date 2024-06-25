import sys
import os
sys.path.append("../Metrics")

import pandas as pd
from Utility_Metrics import trtr, tstr

num_features = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]

real_data = pd.read_csv("/Users/felixdiederichs/PycharmProjects/Data_Analysis/.venv/Data/real/heart_generation.csv")

os.chdir("/Users/felixdiederichs/PycharmProjects/Data_Analysis/.venv/Data/synthetic/heart")

dataframes = {}

for file_names in os.listdir():
    file_path = os.path.join(file_names)
    dataframes[file_names] = pd.read_csv(file_path)
    synthetic_data = dataframes[file_names]
    print(file_names)
    trtr(real_data, "HeartDisease", num_features)
    tstr(real_data, synthetic_data, "HeartDisease", num_features)