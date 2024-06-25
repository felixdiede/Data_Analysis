import sys
sys.path.append("../Metrics")

import pandas as pd
from Resemblance_Metrics import *
from glob import glob
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder



import os

pd.set_option('display.width', None)
pd.set_option('display.max_rows', None)

cat_features = ["Gender", "ChestPainType", "FastingBS", "RestingECG", "ExerciseAngina", "ST_Slope", "HeartDisease"]
num_features = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]

real_data = pd.read_csv("/Users/felixdiederichs/PycharmProjects/Data_Analysis/.venv/Data/real/heart_generation.csv")

os.chdir("/Users/felixdiederichs/PycharmProjects/Data_Analysis/.venv/Data/synthetic/heart/synthetic_normal")

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

    model_priority = {"tabfairgan": 1, "distcorrgan": 2, "multifairgan": 3, "decaf": 4, "TVAE": 5, "CTGAN": 6}
    lambda_priority = {"02": 1, "04": 2, "06": 3, "08": 4, "1": 5, "15": 6, "2": 7, "5": 8}

    return model_priority.get(model, 99), epoch, lambda_priority.get(lambda_value, 99)

file_names = os.listdir()

sorted_files = sorted(file_names, key=custom_sort)

print(file_names)

"""
for file_names in sorted_files:
    file_path = os.path.join(file_names)
    dataframes[file_names] = pd.read_csv(file_path)

    synthetic_data = dataframes[file_names]

    print(file_names)

    statistical_tests = evaluation_statistical_tests(synthetic_data, real_data, num_features, cat_features)

    scaler = MinMaxScaler()
    real_data_scaled = real_data.copy()
    synthetic_data_scaled = synthetic_data.copy()
    real_data_scaled[num_features] = scaler.fit_transform(real_data_scaled[num_features])
    synthetic_data_scaled[num_features] = scaler.transform(synthetic_data_scaled[num_features])

    distances = evaluation_distances(real_data_scaled, synthetic_data_scaled, num_features)

    matrix = ppc_matrix(real_data, synthetic_data, num_features)

    # normalized_contingency_tables(real_data, dataframes["obesity_ctgan_500.csv"], cat_features)

    data_labelling = data_labelling_analysis(real_data, synthetic_data, num_features, cat_features)

    data = [statistical_tests, distances, matrix]
    data.extend(data_labelling)

    data = pd.DataFrame(data, columns = [file_names])

    all_data = pd.concat([all_data, data], axis=1)

all_data.to_excel("/Users/felixdiederichs/PycharmProjects/Data_Analysis/.venv/Code/Resemblance/Reports/Resemblance_Report_heart.xlsx", index=False)






"""