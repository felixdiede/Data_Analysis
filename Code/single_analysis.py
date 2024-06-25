import sys
sys.path.append("Resemblance/Metrics")
from Resemblance_Metrics import *
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

from sklearn.preprocessing import MinMaxScaler

real_data = pd.read_csv("/Users/felixdiederichs/PycharmProjects/Data_Analysis/.venv/Data/real/heart_generation.csv")
synthetic_data = pd.read_csv("/Users/felixdiederichs/PycharmProjects/Data_Analysis/.venv/Data/synthetic/heart/synthetic_normal/heart_decaf_100.csv")

cat_features = ["Gender", "ChestPainType", "FastingBS", "RestingECG", "ExerciseAngina", "ST_Slope", "HeartDisease"]
num_features = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]

def js_distance(real_data, synthetic_data, attribute):
    vector1 = np.array(real_data[attribute])
    vector2 = np.array(synthetic_data[attribute])

    kde1 = gaussian_kde(vector1)
    kde2 = gaussian_kde(vector2)

    xmin = min(vector1.min(), vector2.min())
    xmax = max(vector1.max(), vector2.max())
    x = np.linspace(xmin, xmax, 100)

    p = kde1(x)
    p /= p.sum()
    q = kde2(x)
    q /= q.sum()

    js_dist = jensenshannon(p, q)

    return js_dist

print(js_distance(real_data, synthetic_data, "Oldpeak"))


"""
evaluation_statistical_tests(real_data, synthetic_data, num_features, cat_features)
evaluation_distances(real_data, synthetic_data, num_features)
ppc_matrix(real_data, synthetic_data, num_features)

"""






"""
print("Resemblance Analysis")

evaluation_statistical_tests(synthetic_data, real_data, num_features, cat_features)

scaler = MinMaxScaler()
real_data_scaled = real_data.copy()
synthetic_data_scaled = synthetic_data.copy()
real_data_scaled[num_features] = scaler.fit_transform(real_data_scaled[num_features])
synthetic_data_scaled[num_features] = scaler.transform(synthetic_data_scaled[num_features])

calculate_and_display_distances(real_data_scaled, synthetic_data_scaled, num_features)

ppc_matrix(real_data, synthetic_data, num_features)

# normalized_contingency_tables(real_data, dataframes["obesity_ctgan_500.csv"], cat_features)

data_labelling_analysis(real_data, synthetic_data, num_features, cat_features)

print("Utility Analysis")

trtr(real_data, "NObeyesdad", num_features)
tstr(real_data, synthetic_data, "NObeyesdad", num_features)"""