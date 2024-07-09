import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats
from math import sqrt
from scipy.spatial import distance
from scipy.stats import chi2_contingency
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns

from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split
from univariate_resemblance import *
from multivariate_resemblance import *
from data_labelling import *

real_data = pd.read_csv("/Users/felixdiederichs/PycharmProjects/Data_Analysis/.venv/Data/real/heart_generation.csv")
synthetic_data = pd.read_csv("/Users/felixdiederichs/PycharmProjects/Data_Analysis/.venv/Data/synthetic/heart/heart_1/heart_tabfairgan_50_1.csv")


def execute_t_tests(real_data, synthetic_data, num_features, alpha = 0.05):
    real = real_data[num_features]
    synthetic = synthetic_data[num_features]

    p_values = []

    for c in num_features:
        _, p = stats.ttest_ind(real[c], synthetic[c])
        p_values.append(p)

    positive = 0
    negative = 0
    for val in p_values:
        if val > alpha:
            positive += 1
        else:
            negative += 1

    return p_values, positive, negative

def execute_mann_whitney_tests(real_data, synthetic_data, num_features, alpha = 0.05):
    real = real_data[num_features]
    synthetic = synthetic_data[num_features]

    p_values = []

    # loop to perform the tests for each attribute
    for c in num_features:
        _, p = stats.mannwhitneyu(real[c], synthetic[c])
        p_values.append(p)

    positive = 0
    negative = 0
    for val in p_values:
        if val > alpha:
            positive += 1
        else:
            negative += 1

    return p_values, positive, negative

def execute_ks_tests(real_data, synthetic_data, num_features, alpha = 0.05):
    real = real_data[num_features]
    synthetic = synthetic_data[num_features]

    p_values = []

    # loop to perform the tests for each attribute
    for c in num_features:
        _, p = stats.ks_2samp(real[c], synthetic[c])
        p_values.append(p)

    positive = 0
    negative = 0
    for val in p_values:
        if val > alpha:
            positive += 1
        else:
            negative += 1

    return p_values, positive, negative

def execute_chi_squared_tests(real_data, synthetic_data, cat_features, alpha = 0.05):
    real = real_data[cat_features]
    synthetic = synthetic_data[cat_features]

    p_values = []

    # loop to perform the tests for each attribute
    for c in cat_features:
        # create contingency table
        observed = pd.crosstab(real[c], synthetic[c])
        # perform chi-squared test
        _, p, _, _ = chi2_contingency(observed)
        p_values.append(p)

    positive = 0
    negative = 0
    for val in p_values:
        if val > alpha:
            negative += 1
        else:
            positive += 1

    return p_values, positive, negative

def evaluate_tests(real, synthetic, num_features, cat_features):
    __, positive_t_tests, negative_t_tests = execute_t_tests(real, synthetic, num_features)
    __, positive_mann_whitney_tests, negative_mann_whitney_tests = execute_mann_whitney_tests(real, synthetic, num_features)
    __, positive_ks_tests, negative_ks_tests = execute_ks_tests(real, synthetic, num_features)
    __, positive_chi_squared_tests, negative_chi_squared_tests = execute_chi_squared_tests(real, synthetic, cat_features)

    count_positive_numerical_tests = positive_t_tests + positive_mann_whitney_tests + positive_ks_tests
    count_negative_numerical_tests = negative_t_tests + negative_mann_whitney_tests + negative_ks_tests

    proportion_positive_numerical_tests = count_positive_numerical_tests / (count_positive_numerical_tests + count_negative_numerical_tests)

    count_positive_categorical_tests = positive_chi_squared_tests
    count_negative_categorical_tests = negative_chi_squared_tests

    proportion_positive_categorical_tests = count_positive_categorical_tests / (count_positive_categorical_tests + count_negative_categorical_tests)

    return proportion_positive_numerical_tests, proportion_positive_categorical_tests



def execute_cosine_distances(real_data, synthetic_data):
    dists = cosine_distances(real_data, synthetic_data)

    positive = 0
    negative = 0
    for val in dists:
        if val < 0.3:
            positive += 1
        else:
            negative += 1

    return dists, positive, negative

def execute_js_distances(real_data, synthetic_data):
    dists = js_distances(real_data, synthetic_data)

    positive = 0
    negative = 0
    for val in dists:
        if val < 0.1:
            positive += 1
        else:
            negative += 1

    return dists, positive, negative


def execute_wass_distances(real_data, synthetic_data):
    dists = wass_distances(real_data, synthetic_data)

    positive = 0
    negative = 0
    for val in dists:
        if val < 0.3:
            positive += 1
        else:
            negative += 1

    return dists, positive, negative

def evaluate_distances(real_data, synthetic_data, num_features):
    real_data_scaled = scale_data(real_data[num_features])
    synthetic_data_scaled = scale_data(synthetic_data[num_features])

    __, positive_cosine_distances, negative_cosine_distances = execute_cosine_distances(real_data_scaled, synthetic_data_scaled)
    __, positive_js_distances, negative_js_distances = execute_js_distances(real_data_scaled, synthetic_data_scaled)
    __, positive_wass_distances, negative_wass_distances = execute_wass_distances(real_data_scaled, synthetic_data_scaled)

    count_positive_distances = positive_cosine_distances + positive_js_distances + positive_wass_distances
    count_negative_distances = negative_cosine_distances + negative_js_distances + negative_wass_distances

    proportion_positive_distances = count_positive_distances / (count_positive_distances + count_negative_distances)

    return proportion_positive_distances



def evaluate_correlations(real_data, synthetic_data, num_features, cat_features):
    real_data_numerical_correlations, __ = get_numerical_correlations(real_data[num_features])
    synthetic_data_numerical_correlations, __ = get_numerical_correlations(synthetic_data[num_features])

    real_data_categorical_correlations, __ = get_categorical_correlations(real_data[cat_features])
    synthetic_data_categorical_correlations, __ = get_categorical_correlations(synthetic_data[cat_features])

    mra_score_numerical = compute_mra_score(real_data_numerical_correlations, synthetic_data_numerical_correlations)
    mra_score_categorical = compute_mra_score(real_data_categorical_correlations, synthetic_data_categorical_correlations)

    return mra_score_numerical, mra_score_categorical



def execute_data_labelling(real_data, synthetic_data, num_features, cat_features):
    # Label real and synthetic data
    real_data["label"] = 0
    synthetic_data["label"] = 1

    # Combine both datasets to one
    combined_df = pd.concat([real_data, synthetic_data], axis=0)


    for col in cat_features:
        dummies = pd.get_dummies(combined_df[col], prefix=col)
        combined_df = pd.concat([combined_df.drop(col, axis=1), dummies], axis=1)

    scaler = StandardScaler()
    combined_df[num_features] = scaler.fit_transform(combined_df[num_features])

    # Create combined feature and target dataset
    X = combined_df.drop(columns="label")
    y = combined_df["label"]

    # Split combined dataset into train and test instances
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)

        classifiers = [
            RandomForestClassifier(n_estimators=100, n_jobs=3, random_state=9),
            KNeighborsClassifier(n_neighbors=10, n_jobs=3),
            DecisionTreeClassifier(random_state=9),
            SVC(C=100, max_iter=300, kernel="linear", probability=True, random_state=9),
            MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=300, random_state=9)
        ]

        results = {}
        for clf in classifiers:
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            results[clf.__class__.__name__] = classification_report(y_test, y_pred, output_dict=True)

    results_df = pd.DataFrame(results).transpose()

    average_accuracy = round(results_df["accuracy"].mean(), 4)
    highest_accuracy = round(max(results_df["accuracy"]), 4)
    lowest_accuracy = round(min(results_df["accuracy"]), 4)

    return highest_accuracy, average_accuracy, lowest_accuracy












