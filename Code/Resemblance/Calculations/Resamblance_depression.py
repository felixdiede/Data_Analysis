import sys
sys.path.append("../Metrics")

import pandas as pd
import os
from Resemblance import *
from glob import glob
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

cat_features = ["Sex", "Race", "Housing", "Delay"]
num_features = ["Anhedonia", "Dep_Mood", "Sleep", "Tired", "Appetite", "Rumination", "Concentration", "Psychomotor", "Delusion", "Suspicious", "Withdrawal", "Passive", "Tension", "Unusual_Thoughts"]

real_data = pd.read_csv("/Users/felixdiederichs/PycharmProjects/Data_Analysis/.venv/Data/real/depression_generation.csv")

os.chdir("/Users/felixdiederichs/PycharmProjects/Data_Analysis/.venv/Data/synthetic/depression/synthetic_normal")

all_results = pd.DataFrame()

dataframes = {}

for file_names in os.listdir():
    file_path = os.path.join(file_names)
    dataframes[file_names] = pd.read_csv(file_path)

    synthetic_data = dataframes[file_names]

    statistical_tests = evaluation_statistical_tests(synthetic_data, real_data, num_features, cat_features)

    scaler = MinMaxScaler()
    real_data_scaled = real_data.copy()
    synthetic_data_scaled = synthetic_data.copy()
    real_data_scaled[num_features] = scaler.fit_transform(real_data_scaled[num_features])
    synthetic_data_scaled[num_features] = scaler.transform(synthetic_data_scaled[num_features])

    distances = calculate_and_display_distances(real_data_scaled, synthetic_data_scaled, num_features)

    matrix = ppc_matrix(real_data, synthetic_data, num_features)

    # normalized_contingency_tables()

    data_labelling = data_labelling_analysis(real_data, synthetic_data, num_features, cat_features)

    results = [statistical_tests, distances, matrix]
    results.extend(data_labelling)

    results = pd.DataFrame(results, columns=[file_names])

    all_results = pd.concat([all_results, results], axis=1)

all_results.to_excel("/Users/felixdiederichs/PycharmProjects/Data_Analysis/.venv/Code/Resemblance/Reports/depression", index=False)