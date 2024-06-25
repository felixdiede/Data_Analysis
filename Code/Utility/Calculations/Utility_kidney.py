import sys
import os
sys.path.append("../Metrics")

import warnings
warnings.filterwarnings('always')  # "error", "ignore", "always", "default", "module" or "once"

import pandas as pd
from Utility_Metrics import trtr, tstr


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
    trtr(real_data, "Diagnosis", num_features)
    tstr(real_data, synthetic_data, "Diagnosis", num_features)