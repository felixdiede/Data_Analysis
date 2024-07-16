import os
import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdv.evaluation.single_table import get_column_plot

os.chdir("/Users/felixdiederichs/PycharmProjects/Data_Analysis/.venv/Data")

real_data = pd.read_csv("/Users/felixdiederichs/PycharmProjects/Data_Analysis/.venv/Data/real/obesity_generation.csv")

synthetic_data = pd.read_csv("/Users/felixdiederichs/PycharmProjects/TabFairGAN/.venv/TabFairGAN/experiments/obesity/obesity_1/loop_3/obesity_tabfairgan_250_08.csv")

metadata = SingleTableMetadata()
metadata.detect_from_dataframe(real_data)

fig = get_column_plot(
    real_data=real_data,
    synthetic_data=synthetic_data,
    metadata=metadata,
    column_name="FAVC"
)

fig.show()
