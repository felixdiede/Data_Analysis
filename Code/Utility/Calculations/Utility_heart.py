import sys
import os
sys.path.append("../Metrics")

import psutil
import pandas as pd
from Utility_Metrics import trtr, tstr

num_features = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]

real_data = pd.read_csv("/Users/felixdiederichs/PycharmProjects/Data_Analysis/.venv/Data/real/heart_generation.csv")

os.chdir("/Users/felixdiederichs/PycharmProjects/Data_Analysis/.venv/Data/synthetic/heart/heart_1/loop_10")


def custom_sort(file_name):
    if file_name.endswith(".csv"):
        file_name = file_name[:-4]  # Entferne ".csv"

    parts = file_name.split("_")
    model = parts[1]

    # Überprüfen, ob eine Lambda-Angabe vorhanden ist
    if len(parts) == 4:
        epoch = int(parts[2])
        lambda_value = parts[3]
    else:  # Kein Lambda vorhanden
        epoch = int(parts[2])
        lambda_value = ""  # Leerer String für Lambda

    model_priority = {"tabfairgan": 1, "distcorrgan": 2, "multifairgan": 3, "decaf": 4, "tvae": 5, "ctgan": 6}
    lambda_priority = {"02": 1, "04": 2, "06": 3, "08": 4, "1": 5, "15": 6, "2": 7, "5": 8}

    return model_priority.get(model, 99), epoch, lambda_priority.get(lambda_value, 99)

file_names = os.listdir()
sorted_files = sorted(file_names, key=custom_sort)

dataframes = {}
results = pd.DataFrame()
all_results = pd.DataFrame()

trtr_values = trtr(real_data, "HeartDisease", num_features)

for file_names in sorted_files:
    file_path = os.path.join(file_names)
    dataframes[file_names] = pd.read_csv(file_path)
    synthetic_data = dataframes[file_names]
    print(file_names)
    values = trtr_values

    results = pd.DataFrame(values, columns = [file_names])

    all_results = pd.concat([all_results, results], axis = 1)

all_results.to_csv("/Users/felixdiederichs/PycharmProjects/Data_Analysis/.venv/Reports/heart/Utility/Report_heart_1_1_trtr.csv", index=False)

dataframes = {}
results = pd.DataFrame()
all_results = pd.DataFrame()

for file_names in sorted_files:
    file_path = os.path.join(file_names)
    dataframes[file_names] = pd.read_csv(file_path)
    synthetic_data = dataframes[file_names]
    print(file_names)
    values = tstr(real_data, synthetic_data, "HeartDisease", num_features)

    results = pd.DataFrame(values, columns = [file_names])

    all_results = pd.concat([all_results, results], axis = 1)

all_results.to_csv("/Users/felixdiederichs/PycharmProjects/Data_Analysis/.venv/Reports/heart/Utility/heart_1/loop_10/Report_heart_1_10_tstr.csv", index=False)