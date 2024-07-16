import sys
sys.path.append("../Metrics")

import pandas as pd
from resemblance_wrapper import *
import os

pd.set_option('display.width', None)
pd.set_option('display.max_rows', None)

real_data = pd.read_csv("/Users/felixdiederichs/PycharmProjects/Data_Analysis/.venv/Data/real/obesity_generation.csv")

os.chdir('/Users/felixdiederichs/PycharmProjects/Data_Analysis/.venv/Data/synthetic/obesity/obesity_1/loop_10')

cat_features = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS', 'NObeyesdad']
num_features = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', "CH2O", 'FAF', 'TUE']

int_features = []
float_features = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']


for col in cat_features:
    real_data[col] = real_data[col].astype("category")
for col in int_features:
    real_data[col] = real_data[col].astype("int64")
for col in float_features:
    real_data[col] = real_data[col].astype("float64")

print(f"Numerical columns: {real_data.select_dtypes(include=['int64', 'float64'])}")
print(f"Categorical columns: {real_data.select_dtypes(include=['category'])}")

all_data = pd.DataFrame()
dataframes = {}

def custom_sort(file_name):
    if file_name.endswith(".csv"):
        file_name = file_name[:-4]

    parts = file_name.split("_")
    model = parts[1]

    if len(parts) == 4:
        epoch = int(parts[2])
        lambda_value = parts[3]
    else:
        epoch = int(parts[2])
        lambda_value = ""

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

all_data.to_excel('/Users/felixdiederichs/PycharmProjects/Data_Analysis/.venv/Reports/obesity/Resamblance/Report_resemblance_obesity_1_10.xlsx', index=False)
