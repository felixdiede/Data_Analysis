import sys
sys.path.append("../Metrics")

import pandas as pd
from resemblance_wrapper import *
from glob import glob
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

import os

pd.set_option('display.width', None)
pd.set_option('display.max_rows', None)

cat_features = ["Gender", "ChestPainType", "FastingBS", "RestingECG", "ExerciseAngina", "ST_Slope", "HeartDisease"]
num_features = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]

int_features = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
float_features = ["Oldpeak"]

real_data = pd.read_csv("/Users/felixdiederichs/PycharmProjects/Data_Analysis/.venv/Data/real/heart_generation.csv")

real_data[cat_features] = real_data[cat_features].astype("category")
print(f"Numerical columns: {real_data.select_dtypes(include=['int64', 'float64'])}")
print(f"Categorical columns: {real_data.select_dtypes(include=['category'])}")

os.chdir("/Users/felixdiederichs/PycharmProjects/Data_Analysis/.venv/Data/synthetic/heart/heart_0/loop_1")

all_data = pd.DataFrame()
dataframes = {}

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

for file_names in sorted_files:
    file_path = os.path.join(file_names)
    dataframes[file_names] = pd.read_csv(file_path)


    synthetic_data = dataframes[file_names]
    print(file_names)

    for col in cat_features:
        synthetic_data[col] = synthetic_data[col].astype("category")
    for col in int_features:
        synthetic_data[col] = synthetic_data[col].astype("int64")
    for col in float_features:
        synthetic_data[col] = synthetic_data[col].astype("float64")

    results_tests = evaluate_tests(real_data, synthetic_data, num_features, cat_features)
    results_distances = evaluate_distances(real_data, synthetic_data, num_features)
    results_correlations = evaluate_correlations(real_data, synthetic_data, num_features, cat_features)
    results_data_labelling = execute_data_labelling(real_data, synthetic_data, num_features, cat_features)

    results = []
    results.extend(results_tests)
    results.append(results_distances)
    results.extend(results_correlations)
    results.extend(results_data_labelling.values)

    results = pd.DataFrame(results, columns = [file_names])

    all_data = pd.concat([all_data, results], axis=1)

all_data.to_excel("/Users/felixdiederichs/PycharmProjects/Data_Analysis/.venv/Reports/heart/Resamblance/Report_resemblance_heart_0_1.xlsx", index=False)






