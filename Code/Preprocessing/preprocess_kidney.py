import pandas as pd
import os

os.chdir("/Users/felixdiederichs/PycharmProjects/Data_Analysis/.venv")

data = pd.read_csv("Data/real/kidney.csv")

data.drop(["PatientID", "DoctorInCharge"], inplace=True, axis=1)

data["EducationLevel"] = data["EducationLevel"].apply(lambda x: 0 if x==3 else 1)

data.to_csv("Data/real/kidney_generation.csv", index=False)