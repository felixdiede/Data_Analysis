import sys
import os
sys.path.append("../Metrics")

import pandas as pd
from Utility_Metrics import trtr, tstr

num_features = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"]

real_data = pd.read_csv("/Users/felixdiederichs/PycharmProjects/Data_Analysis/.venv/Data/real/obesity_generation.csv")

directory_path = "/Users/felixdiederichs/PycharmProjects/Data_Analysis/.venv/Data/synthetic/obesity/obesity_1"

report_path = "/Users/felixdiederichs/PycharmProjects/Data_Analysis/.venv/Reports"

os.chdir(directory_path)

trtr_values = trtr(real_data, "NObeyesdad", num_features)

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

def custom_sort_2(loop_name):
    parts = loop_name.split("_")
    number = parts[1]

    number_priority = {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "10": 10}

    return number_priority.get(number, 99)


for loop in sorted(os.listdir(directory_path), key=custom_sort_2):

    loop_path = f"/Users/felixdiederichs/PycharmProjects/Data_Analysis/.venv/Data/synthetic/obesity/obesity_1/{loop}"

    os.chdir(loop_path)

    for i in os.listdir(loop_path):

        file_names = os.listdir()
        sorted_files = sorted(file_names, key=custom_sort)

        dataframes = {}
        results = pd.DataFrame()
        all_results = pd.DataFrame()

        for file_names in sorted_files:
            file_path = os.path.join(file_names)
            dataframes[file_names] = pd.read_csv(file_path)
            synthetic_data = dataframes[file_names]
            print(file_names)
            values = trtr_values

            results = pd.DataFrame(values, columns=[file_names])

            all_results = pd.concat([all_results, results], axis=1)

        all_results.to_csv(f"{report_path}/obesity/Utility/obesity_1/{loop}/utility_obesity_1_{loop}_trtr.csv", index=False)

        dataframes = {}
        results = pd.DataFrame()
        all_results = pd.DataFrame()

        for file_names in sorted_files:
            print(f"Loop: {loop}, Datei: {file_names}")
            file_path = os.path.join(file_names)
            dataframes[file_names] = pd.read_csv(file_path)
            synthetic_data = dataframes[file_names]

            values = tstr(real_data, synthetic_data, "NObeyesdad", num_features)

            results = pd.DataFrame(values, columns=[file_names])

            all_results = pd.concat([all_results, results], axis=1)

        all_results.to_csv(f"{report_path}/obesity/Utility/obesity_1/{loop}/obesity_1_{loop}_tstr.csv",index=False)



