import pandas as pd
import os

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

os.chdir("/Users/felixdiederichs/PycharmProjects/Data_Analysis/.venv")

kidney = pd.read_csv("Data/real/kidney.csv")

kidney_synthetic = pd.read_csv("Data/synthetic/kidney/kidney_decaf.csv")

print(kidney.describe())
print(kidney.info())

print(kidney_synthetic.describe())
print(kidney.info())