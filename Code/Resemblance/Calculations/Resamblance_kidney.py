import sys
sys.path.append("../Metrics")

import pandas as pd
from Resemblance_Metrics import *
from glob import glob
from sklearn.preprocessing import MinMaxScaler

import os

pd.set_option('display.width', None)
pd.set_option('display.max_rows', None)


num_features = ["Age", "BMI", "AlcoholConsumption", "PhysicalActivity",
                "DietQuality", 'AlcoholConsumption', 'PhysicalActivity',
                'DietQuality', 'SleepQuality', 'SystolicBP',
                'DiastolicBP', 'FastingBloodSugar', 'HbA1c', 'SerumCreatinine',
                'BUNLevels', 'GFR', 'ProteinInUrine', 'ACR', 'SerumElectrolytesSodium',
                'SerumElectrolytesPotassium', 'SerumElectrolytesCalcium',
                'SerumElectrolytesPhosphorus', 'HemoglobinLevels', 'CholesterolTotal',
                'CholesterolLDL', 'CholesterolHDL', 'CholesterolTriglycerides', "NSAIDsUse",
                'FatigueLevels', 'NauseaVomiting',
                'MuscleCramps', 'Itching', "QualityOfLifeScore",
                'MedicalCheckupsFrequency', 'MedicationAdherence', 'HealthLiteracy'
                ]

cat_features = ["Gender", "Ethnicity", "SocioeconomicStatus", "EducationLevel", "Smoking", "FamilyHistoryKidneyDisease", "FamilyHistoryHypertension",
                "FamilyHistoryDiabetes", "PreviousAcuteKidneyInjury", "UrinaryTractInfections", "ACEInhibitors", "Diuretics", "Statins", "AntidiabeticMedications",
                "Edema", "HeavyMetalsExposure", "OccupationalExposureChemicals", "WaterQuality", "Diagnosis"]

real_data = pd.read_csv("/Users/felixdiederichs/PycharmProjects/Data_Analysis/.venv/Data/real/kidney_generation.csv")
os.chdir("/Users/felixdiederichs/PycharmProjects/Data_Analysis/.venv/Data/synthetic/kidney")

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

    data_labelling_analysis(real_data, synthetic_data, num_features, cat_features)