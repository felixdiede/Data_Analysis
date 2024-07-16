import pandas as pd

data = pd.read_csv("/Users/felixdiederichs/PycharmProjects/Data_Analysis/.venv/Data/real/depression_original.csv")

data.drop("dataset", axis=1, inplace=True)

data["Race"] = data["Race"].apply(lambda x: 0 if x=="Black" else 1)

# data = pd.get_dummies(data)

#data = data.sample(frac = 0.5, random_state = 0, replace = True)

data.to_csv("/Users/felixdiederichs/PycharmProjects/Data_Analysis/.venv/Data/real/depression_generation_not_short.csv", index=False)