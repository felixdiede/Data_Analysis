import pandas as pd

data = pd.read_csv("/Users/felixdiederichs/PycharmProjects/Data_Analysis/.venv/Data/real/diabetes_original.csv")

data = data.sample(n = 7500, random_state = 0)
data["Diabetes_binary"] = data["Diabetes_binary"].astype(int)

data["Income"] = data["Income"].apply(lambda x: 0 if x < 4 else 1)

data.to_csv("/Users/felixdiederichs/PycharmProjects/Data_Analysis/.venv/Data/real/diabetes_generation.csv", index = False)
