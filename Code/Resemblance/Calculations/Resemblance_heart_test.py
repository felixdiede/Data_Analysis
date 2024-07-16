import sys
sys.path.append("../Metrics")

import pandas as pd
from resemblance_wrapper_test import *
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

os.chdir("/Users/felixdiederichs/PycharmProjects/Data_Analysis/.venv/Data/synthetic/heart/heart_1/loop_1")

all_distances = pd.DataFrame()
all_categorical_tests = pd.DataFrame()
all_numerical_tests = pd.DataFrame()

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

def lambda_to_decimal(lambda_value):
    try:
        if lambda_value:
            # Konvertiere nur Lambda-Werte mit führenden Nullen
            if lambda_value in ["02", "04", "06", "08", '15']:
                return str(float(int(lambda_value) / 10))
            else:
                # Belasse bereits korrekte Dezimalwerte unverändert
                return lambda_value
        else:
            return ""
    except ValueError:
        return ""

file_names = os.listdir()

sorted_files = sorted(file_names, key=custom_sort)

loop = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

for i in loop:
    print(f"Loop: {i}")
    for file_names in sorted_files:
        os.chdir(f'/Users/felixdiederichs/PycharmProjects/Data_Analysis/.venv/Data/synthetic/heart/heart_1/loop_{i}')

        file_path = os.path.join(file_names)
        dataframes[file_names] = pd.read_csv(file_path)

        synthetic_data = dataframes[file_names]

        for col in cat_features:
            synthetic_data[col] = synthetic_data[col].astype("category")
        for col in int_features:
            synthetic_data[col] = synthetic_data[col].astype("int64")
        for col in float_features:
            synthetic_data[col] = synthetic_data[col].astype("float64")

        file_names = file_names[:-4]
        parts = file_names.split("_")
        model = parts[1]
        epochs = parts[2]
        if len(parts) == 4:
            lambda_value = parts[3]

        results_numerical_tests_df = pd.DataFrame(evaluate_numerical_tests(real_data, synthetic_data, num_features))
        results_categorical_tests_df = pd.DataFrame(evaluate_categorical_tests(real_data, synthetic_data, cat_features))
        results_distances_df = pd.DataFrame(results_distances(real_data, synthetic_data, num_features))

        results_numerical_tests_df['Loop'] = int(i)
        results_distances_df['Loop'] = int(i)
        results_numerical_tests_df['Model'] = model
        results_distances_df['Model'] = model
        results_numerical_tests_df['Epochs'] = int(epochs)
        results_distances_df['Epochs'] = int(epochs)
        if model not in ['tvae', 'ctgan']:
            lambda_decimal = lambda_to_decimal(lambda_value)
            results_numerical_tests_df['Lambda'] = float(lambda_decimal)
            results_distances_df['Lambda'] = float(lambda_decimal)

        all_numerical_tests = pd.concat([all_numerical_tests, results_numerical_tests_df], axis=0)
        all_categorical_tests = pd.concat([all_categorical_tests, results_categorical_tests_df], axis=0)
        all_distances = pd.concat([all_distances, results_distances_df], axis=0)

    all_numerical_tests = all_numerical_tests
    all_categorical_tests = all_categorical_tests
    all_distances = all_distances

repeated_list = num_features * 1560

all_numerical_tests['Variable'] = repeated_list
all_distances['Variable'] = repeated_list

all_numerical_tests.to_excel('/Users/felixdiederichs/PycharmProjects/Data_Analysis/.venv/Reports/heart/Resamblance/Report_heart_resemblance_numerical_tests.xlsx', index=False)
all_categorical_tests.to_excel('/Users/felixdiederichs/PycharmProjects/Data_Analysis/.venv/Reports/heart/Resamblance/Report_heart_resemblance_categorical_tests.xlsx', index=False)
all_distances.to_excel('/Users/felixdiederichs/PycharmProjects/Data_Analysis/.venv/Reports/heart/Resamblance/Report_heart_resemblance_distances.xlsx', index=False)






