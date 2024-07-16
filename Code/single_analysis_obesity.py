import sys
sys.path.append("Resemblance/Metrics")
from Resemblance_Metrics import *
from resemblance_wrapper import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import ttest_ind, mannwhitneyu, chi2_contingency, gaussian_kde, entropy
from scipy.spatial.distance import jensenshannon
from sklearn.metrics.pairwise import cosine_distances
from scipy import spatial

from scipy.special import kl_div
from scipy import stats
from scipy.stats import entropy, wasserstein_distance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
from sklearn.exceptions import ConvergenceWarning

sys.path.append("Utility/Metrics")
from Utility_Metrics import *
import os

pd.set_option('display.width', None)
pd.set_option('display.max_rows', None)

os.chdir("/Users/felixdiederichs/PycharmProjects/Data_Analysis/.venv/Data")
real_data = pd.read_csv("real/obesity_generation.csv")
synthetic_data = pd.read_csv("/Users/felixdiederichs/PycharmProjects/Data_Analysis/.venv/Data/synthetic/obesity/obesity_1/loop_1/obesity_tabfairgan_50_02.csv")

cat_features = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS', 'NObeyesdad']
num_features = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', "CH2O", 'FAF', 'TUE']

int_features = []
float_features = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']

for col in cat_features:
    synthetic_data[col] = synthetic_data[col].astype("category")
for col in int_features:
    synthetic_data[col] = synthetic_data[col].astype("int64")
for col in float_features:
    synthetic_data[col] = synthetic_data[col].astype("float64")

for col in cat_features:
    real_data[col] = real_data[col].astype("category")
for col in int_features:
    real_data[col] = real_data[col].astype("int64")
for col in float_features:
    real_data[col] = real_data[col].astype("float64")

print(real_data.dtypes)
print(synthetic_data.dtypes)

t_test_p_values, t_test_positive, t_test_negative = execute_t_tests(real_data, synthetic_data, num_features)
print(f"t-test: {t_test_p_values, t_test_positive, t_test_negative}")

mw_test_p_values, mw_test_positive, mw_test_negative = execute_mann_whitney_tests(real_data, synthetic_data, num_features)
print(f"mw-test: {mw_test_p_values, mw_test_positive, mw_test_negative}")

ks_test_p_values, ks_test_positive, ks_test_negative = execute_ks_tests(real_data, synthetic_data, num_features)
print(f"ks-test: {ks_test_p_values, ks_test_positive, ks_test_negative}")

chi_test_p_values, chi_test_positive, chi_test_negative = execute_chi_squared_tests(real_data, synthetic_data, cat_features)
print(f"chi-test: {chi_test_p_values, chi_test_positive, chi_test_negative}")

distances = evaluate_distances(real_data, synthetic_data, num_features)
print(f"distances: {distances}")

corr_num, corr_cat = evaluate_correlations(real_data, synthetic_data, num_features, cat_features)

print(f"corr: {corr_num, corr_cat}")