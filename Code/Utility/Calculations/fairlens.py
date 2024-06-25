import fairlens as fl
import pandas as pd

data = pd.read_csv("/Users/felixdiederichs/PycharmProjects/Data_Analysis/.venv/Data/real/diabetes.csv")

fs = fl.FairnessScorer(data, "Diabetes_binary")
fs.demographic_report()