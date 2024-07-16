import pandas as pd
from plotnine import *

gen_1_report = pd.read_csv("/Users/felixdiederichs/PycharmProjects/Data_Analysis/.venv/Reports/obesity/Fairness/Report_fairness_obesity_1_1.csv")
gen_2_report = pd.read_csv("/Users/felixdiederichs/PycharmProjects/Data_Analysis/.venv/Reports/obesity/Fairness/Report_fairness_obesity_1_2.csv")
gen_3_report = pd.read_csv("/Users/felixdiederichs/PycharmProjects/Data_Analysis/.venv/Reports/obesity/Fairness/Report_fairness_obesity_1_3.csv")
gen_4_report = pd.read_csv("/Users/felixdiederichs/PycharmProjects/Data_Analysis/.venv/Reports/obesity/Fairness/Report_fairness_obesity_1_4.csv")
gen_5_report = pd.read_csv("/Users/felixdiederichs/PycharmProjects/Data_Analysis/.venv/Reports/obesity/Fairness/Report_fairness_obesity_1_5.csv")

gen_1_report = gen_1_report.iloc[:, 48:]

spd_1 = gen_1_report.iloc[8]
spd_2 = gen_2_report.iloc[8]
spd_3 = gen_3_report.iloc[8]
spd_4 = gen_4_report.iloc[8]
spd_5 = gen_5_report.iloc[8]

df = pd.concat([spd_1, spd_2, spd_3, spd_4, spd_5], axis=1)
df.columns = ["Gen_1", "Gen_2", "Gen_3", "Gen_4", "Gen_5"]

df["x"] = df.index

df_long = df.melt(id_vars='x', var_name='Gene', value_name='Value')

df_long['x'] = pd.to_numeric(df_long['x'], errors='coerce')
df_long = df_long.dropna(subset=['x'])

p = ggplot(df_long, aes(x="x",y="Value", color="Gene")) + geom_line()
p.show()