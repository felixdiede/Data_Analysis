import matplotlib.pyplot as plt
import pandas as pd
import fairlens as fl
import seaborn as sns

real_data = pd.read_csv('/Users/felixdiederichs/PycharmProjects/Data_Analysis/.venv/Data/real/heart_generation.csv')
synth_data = pd.read_csv("/Users/felixdiederichs/PycharmProjects/Data_Analysis/.venv/Data/synthetic/heart/heart_decaf.csv")

sns.pairplot(real_data)
plt.show()

sns.pairplot(synth_data)
plt.show()