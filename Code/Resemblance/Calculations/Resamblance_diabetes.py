import sys
sys.path.append("../Metrics")

import pandas as pd
from Resemblance_Metrics import *
from glob import glob
from sklearn.preprocessing import MinMaxScaler

import os

pd.set_option('display.width', None)
pd.set_option('display.max_rows', None)

cat_features = ["Diabetes_binary", "HighBP", "HighChol", "CholCheck", "Smoker", "Stroke", "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth", "DiffWalk", "Sex", "Education", "Income"]
num_features = ["BMI", "MentHlth", "PhysHlth", "Age"]

real_data = pd.read_csv("/Users/felixdiederichs/PycharmProjects/Data_Analysis/.venv/Data/real/diabetes_generation.csv")

output_dir = "/Users/felixdiederichs/PycharmProjects/Data_Analysis/.venv/Reports/diabetes/Resamblance"

os.chdir("/Users/felixdiederichs/PycharmProjects/Data_Analysis/.venv/Data/synthetic/diabetes/synthetic_normal")

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

        categorical_statistical_tests = evaluation_categorical_statistical_tests(real_data, synthetic_data, cat_features)
        numerical_statistical_tests = evaluation_numerical_statistical_tests(real_data, synthetic_data, num_features)

        scaler = MinMaxScaler()
        real_data_scaled = real_data.copy()
        synthetic_data_scaled = synthetic_data.copy()
        real_data_scaled[num_features] = scaler.fit_transform(real_data_scaled[num_features])
        synthetic_data_scaled[num_features] = scaler.transform(synthetic_data_scaled[num_features])

        distances = evaluation_distances(real_data_scaled, synthetic_data_scaled, num_features)

        matrix = ppc_matrix(real_data, synthetic_data, num_features)

        # normalized_contingency_tables(real_data, dataframes["obesity_ctgan_500.csv"], cat_features)

        data_labelling = data_labelling_analysis(real_data, synthetic_data, num_features)

        results = [categorical_statistical_tests, numerical_statistical_tests, distances, matrix]
        results.extend(data_labelling)

        results = pd.DataFrame(results, columns = [file_names])

        output_file = os.path.join(output_dir, f"{file_names}_intermediate_results.csv")
        results.to_csv(output_file, index=False)

        all_data = pd.concat([all_data, results], axis=1)

all_data.to_csv("/Users/felixdiederichs/PycharmProjects/Data_Analysis/.venv/Reports/diabetes/Resamblance/Report_diabetes_resamblance.csv", index = False)
