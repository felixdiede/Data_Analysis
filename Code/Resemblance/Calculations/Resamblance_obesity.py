import sys
sys.path.append("../Metrics")

import pandas as pd
from Resemblance_Metrics import *
from glob import glob
from sklearn.preprocessing import MinMaxScaler

import os

pd.set_option('display.width', None)
pd.set_option('display.max_rows', None)



cat_features = ["Gender", "family_history_with_overweight", "FAVC", "CAEC", "SMOKE", "SCC", "CALC", "MTRANS", "NObeyesdad"]
num_features = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"]

real_data = pd.read_csv("/Users/felixdiederichs/PycharmProjects/Data_Analysis/.venv/Data/real/obesity_generation.csv")

os.chdir("/Users/felixdiederichs/PycharmProjects/Data_Analysis/.venv/Data/synthetic/obesity")

dataframes = {}

for file_names in os.listdir():
        file_path = os.path.join(file_names)
        dataframes[file_names] = pd.read_csv(file_path)

        synthetic_data = dataframes[file_names]

        print(file_names)

        evaluation_statistical_tests(synthetic_data, real_data, num_features, cat_features)

        scaler = MinMaxScaler()
        real_data_scaled = real_data.copy()
        synthetic_data_scaled = synthetic_data.copy()
        real_data_scaled[num_features] = scaler.fit_transform(real_data_scaled[num_features])
        synthetic_data_scaled[num_features] = scaler.transform(synthetic_data_scaled[num_features])

        calculate_and_display_distances(real_data_scaled, synthetic_data_scaled, num_features)

        ppc_matrix(real_data, synthetic_data, num_features)

        # normalized_contingency_tables(real_data, dataframes["obesity_ctgan_500.csv"], cat_features)

        data_labelling_analysis(real_data, synthetic_data, num_features, cat_features)


