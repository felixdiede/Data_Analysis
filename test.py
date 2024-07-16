import pandas as pd
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv('/Users/felixdiederichs/PycharmProjects/Data_Analysis/.venv/Reports/heart/Utility/heart_1/loop_1/Report_heart_1_1_trtr.csv')

data.to_excel('/Users/felixdiederichs/PycharmProjects/Data_Analysis/.venv/Reports/heart/Utility/heart_1/loop_1/Report_heart_trtr.xlsx', index=False)
