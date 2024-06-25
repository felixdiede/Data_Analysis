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


"""
Chapter 1: Univariate Resemblance Analysis
    1.1 Statistical test for numerical attributes
        1.1.1 Student T-test for the comparison of means
        1.1.2 Mann Whitney U-test for population comparison
        1.1.3 Kolmogorov-Smirnov test for distribution comparison
"""
def t_test(real_data: pd.DataFrame, synthetic_data: pd.DataFrame, attribute: str, alpha=0.05):
    """
    Implementation of the two-sample t-test. In this case, two independent samples with unequal variances are assumed (so-called "Welch-Test").
    This statistical test evaluates if two independent samples have identical average expected values.
    During calculation, the ttest_ind()-function from the scipy.stats-module was used.
    --> H0: two independent samples have identical average expected values.
    --> H1: two independent samples do not have identical average expected values.
    Conclusion:
    If the calculated p-value is smaller than alpha, H0 is rejected. In this case, there is a statistically significant difference between the means of the two samples.
    If the calculated p-value is greater than alpha, H0 is not rejected. In this case, there is no statistically significant difference between the means of the two samples.

    In the end, the rejection or non-rejection of H0 is evaluated in the context of resemblance. Therefore, if H0 gets
    rejected, the function returns a "negative" conclusion. If H0 does not get rejected and the resemblance is indeed given, the conclusion
    "positive" is returned.

    :param real_data: pandas DataFrame containing the real data
    :param synthetic_data: pandas DataFrame containing the synthetic data
    :param attribute: String that represents the attribute of the data that should be tested
    :param alpha: Level of significance of the given test. Defaults to 0.05.
    :return: Value of the t-statistic; p-Value of the test; Conclusion of the test
    """

    # Check for correct input
    if attribute not in real_data.columns:
        raise ValueError(f"t-test: Attribute {attribute} not found in real data.")
    if attribute not in synthetic_data.columns:
        raise ValueError(f"t-test: Attribute {attribute} not found in synthetic data.")
    if pd.api.types.is_numeric_dtype(real_data[attribute]) == False:
        raise ValueError(f"t-test: Attribute {attribute} of real data is not numeric.")
    if pd.api.types.is_numeric_dtype(synthetic_data[attribute]) == False:
        raise ValueError(f"t-test: Attribute {attribute} of synthetic data is not numeric.")

    # Convert the data currently implemented in a pandas series object into a numpy array
    vector1 = np.array(real_data[attribute])
    vector2 = np.array(synthetic_data[attribute])

    # Calculation of the actual test. The function ttest_ind() itself stems from the "scipy.stats" module
    t_statistic, p_value = ttest_ind(vector1, vector2, equal_var=False, alternative="two-sided")

    # "H0 is rejected, if the p-value is smaller than alpha. Means of synthetic and real data are not equal. For this attribute, the resamblance is not given."
    if p_value < alpha:
        conclusion = "negative"
    # "H0 is not rejected. Means of synthetic and reald data are equal. For this attribute, the resembalance is given."
    else:
        conclusion = "positive"

    return t_statistic, p_value, conclusion


def print_results_t_test(real_data: pd.DataFrame, synthetic_data: pd.DataFrame, attribute):
    positive = 0
    negative = 0

    for attr in attribute:
        t_statistic, p_value, conclusion = t_test(real_data, synthetic_data, attr)

        if conclusion == "positive":
            positive += 1
        else:
            negative += 1

    return positive, negative



def mw_test(real_data: pd.DataFrame, synthetic_data: pd.DataFrame, attribute, alpha=0.05):
    # Check for correct input
    if attribute not in real_data.columns:
        raise ValueError(f"t-test: Attribute {attribute} not found in real data.")
    if attribute not in synthetic_data.columns:
        raise ValueError(f"t-test: Attribute {attribute} not found in synthetic data.")
    if pd.api.types.is_numeric_dtype(real_data[attribute]) == False:
        raise TypeError(f"t-test: Attribute {attribute} of real data is not numeric.")
    if pd.api.types.is_numeric_dtype(synthetic_data[attribute]) == False:
        raise TypeError(f"t-test: Attribute {attribute} of synthetic data is not numeric.")

    vector1 = np.array(real_data[attribute])
    vector2 = np.array(synthetic_data[attribute])

    mw_statistic, p_value = mannwhitneyu(vector1, vector2, alternative="two-sided")

    if p_value < alpha:
        conclusion = "negative"
            #"H0 is rejected. The attribute of the synthetic and real data do not come from the same distribution.
            # For this attribute, the resamblance is not given."
    else:
        conclusion = "positive"
            #"H0 is not rejected. The attribute of the synthetic and real do come from the same distribution.
            # For this attribute, the resemblance is given."

    return mw_statistic, p_value, conclusion



def print_results_mw_test(real_data, synthetic_data, attribute):
    positive = 0
    negative = 0

    for attr in attribute:
        mw_statistic, p_value, conclusion = mw_test(real_data, synthetic_data, attr)

        if conclusion == "positive":
            positive += 1
        else:
            negative += 1

    return positive, negative



def ks_test(real_data, synthetic_data, attribute, alpha=0.05):
    vector1 = np.array(real_data[attribute])
    vector2 = np.array(synthetic_data[attribute])

    ks_statistic, p_value = stats.ks_2samp(vector1, vector2, alternative = "two-sided")

    if p_value < alpha:
        conclusion = "negative"
            #"H0 is rejected. The attribute of the real and synthetic data do not have equal distributions.
            # For this attribute, the resamblance is not given."
    else:
        conclusion = "positive"
            #"H0 is not rejected. the attribute of the real and synthetic data do have equal distributions.
            # For this attribute, the resembalance is given."

    return ks_statistic, p_value, conclusion

def print_results_ks_test(real_data, synthetic_data, attribute):
    positive = 0
    negative = 0

    for attr in attribute:
        ks_statistic, p_value, conclusion = ks_test(real_data, synthetic_data, attr)

        if conclusion == "positive":
            positive += 1
        else:
            negative += 1

    return positive, negative



"""
Chapter 1: Univariate Resemblance Analysis
    1.2 Statistical test for categorical attributes
        1.2.1 Chi-square test 
"""
def chi2_test(real_data, synthetic_data, attribute, alpha=0.05):
    contingency_table = pd.crosstab(real_data[attribute], synthetic_data[attribute])

    chi2, p, dof, expected = chi2_contingency(contingency_table)

    if chi2 < alpha:
        conclusion = "positive"
            # "H0 is rejected. There exists a relationship between the real categorical variable and the synthetic categorical variable.
            # For this attribute, the resembalance is given."
    else:
        conclusion = "negative"
            #"H0 is not rejected. There exists no relationship between the real categorical variable and the synthetic categorical variable.
            # For this attribute, the resamblance is not given."

    return chi2, p, dof, expected, conclusion



def print_results_chi2_test(real_data, synthetic_data, attribute):
    positive = 0
    negative = 0

    for attr in attribute:
        chi2, p, dof, expected, conclusion = chi2_test(real_data, synthetic_data, attr)
        if conclusion == "positive":
            positive += 1
        else:
            negative += 1

    return positive, negative



def evaluation_statistical_tests(real_data, synthetic_data, num_features, cat_features):
    positive_t, negative_t = print_results_t_test(real_data, synthetic_data, num_features)
    positive_mw, negative_mw = print_results_mw_test(real_data, synthetic_data, num_features)
    positive_ks, negative_ks = print_results_ks_test(real_data, synthetic_data, num_features)
    positive_chi2, negative_chi2 = print_results_chi2_test(real_data, synthetic_data, cat_features)

    total_positive_tests = positive_t + positive_mw + positive_ks + positive_chi2
    total_negative_tests = negative_t + negative_mw + negative_ks + negative_chi2
    proportion_positive_tests = total_positive_tests / (total_negative_tests + total_positive_tests)

    return round(proportion_positive_tests, 4)

"""
Chapter 1: Univariate Resemblance Analysis
    1.3 Distance calculation
        1.3.1 Cosine distance
        1.3.2 Jensen-Shannon Distance
        1.3.3 Wassertein Distance
"""
def cos_distance(real_data, synthetic_data, attribute):
    vector1 = np.array([real_data[attribute]])
    vector2 = np.array([synthetic_data[attribute]])

    cos_dist = cosine_distances(vector1, vector2)

    return cos_dist



def results_cos_distance(real_data, synthetic_data, attribute, threshold=0.3):
    positive = 0
    negative = 0

    for attr in attribute:
        cos_dist = cos_distance(real_data, synthetic_data, attr)

        if cos_dist < threshold:
            positive += 1
        else:
            negative += 1

    return positive, negative



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



def results_js_distance(real_data, synthetic_data, attribute, threshold=0.1):
    positive = 0
    negative = 0

    for attr in attribute:
        js_dist = js_distance(real_data, synthetic_data, attr)

        if js_dist < threshold:
            positive += 1
        else:
            negative += 1

    return positive, negative



def was_distance(real_data, synthetic_data, attribute):
    vector1 = np.array(real_data[attribute])
    vector2 = np.array(synthetic_data[attribute])

    u_values = np.histogram(vector1)[0] / len(vector1)
    v_values = np.histogram(vector2)[0] / len(vector2)

    ws_distance = wasserstein_distance(u_values, v_values)

    return ws_distance



def results_was_distance(real_data, synthetic_data, attribute, threshold=0.3):
    positive = 0
    negative = 0

    for attr in attribute:
        was_dist = was_distance(real_data, synthetic_data, attr)

        if was_dist < threshold:
            positive += 1
        else:
            negative += 1

    return positive, negative



def evaluation_distances(real_data, synthetic_data, num_features):
    positive_cos_distance, negative_cos_distance = results_cos_distance(real_data, synthetic_data, num_features)
    positive_js_distance, negative_js_distance = results_js_distance(real_data, synthetic_data, num_features)
    positive_was_distance, negative_was_distance = results_was_distance(real_data, synthetic_data, num_features)

    total_positive_distances = positive_js_distance + positive_cos_distance + positive_was_distance
    total_negative_distances = negative_js_distance + negative_cos_distance + negative_was_distance
    proportion_positive_distances = total_positive_distances / (total_negative_distances + total_positive_distances)

    return round(proportion_positive_distances, 4)










def calculate_and_display_distances(real_data, synthetic_data, attribute):
    thresholds = {
        "Cosinus": 0.3,
        "Jensen-Shannon": 0.1,
       # "KL-Divergenz": 0.1,
        "Wasserstein": 0.3
    }

    distance_functions = {
        "Cosinus": cos_distance,
        "Jensen-Shannon": js_distance,
        # "KL-Divergenz": kl_divergence,
        "Wasserstein": was_distance
    }

    all_results = {}
    total_true_false = {"true": 0, "false": 0}

    for distance_name, distance_func in distance_functions.items():
        results = []
        for attr in attribute:
            try:
                distance = distance_func(real_data, synthetic_data, attr)
                results.append({"Attribute": attr, "Distance": distance})
            except ValueError:
                results.append({"Attribute": attr, "Distance": "N/A (Error)"})

        df = pd.DataFrame(results)
        df["Conclusion"] = df["Distance"].apply(lambda x: "true" if x < thresholds[distance_name] else "false")

        markdown_table = df.to_markdown(index=False, numalign="left", stralign="left")
        all_results[distance_name] = markdown_table

        """
        # Ausgabe der Tabelle und Zusammenfassung direkt darunter
        print(f"\n## {distance_name} Distanzen\n")
        print(markdown_table)"""

        # Zählen von true/false pro Tabelle
        value_counts = df["Conclusion"].value_counts()
        for value, count in value_counts.items():
            total_true_false[value] += count
            """
            print(f"Anzahl '{value}': {count}")

    # Gesamtzusammenfassung am Ende
    print("\n## Gesamtzusammenfassung\n")
    for value, count in total_true_false.items():
        print(f"Gesamt Anzahl '{value}': {count}")"""
    summary = total_true_false.copy()

    return round(summary["true"]/ (summary["true"] + summary["false"]), 4)

"""
Chapter 2: Multivariate Relationship Analysis
    2.1 PPC Matrices comparison
"""

def ppc_matrix(real_data, synthetic_data, num_features, threshold=0.1):


    num_real_data = real_data[num_features]
    num_synthetic_data = synthetic_data[num_features]

    corr_matrix_real = num_real_data.corr()
    corr_matrix_syn = num_synthetic_data.corr()

    diff_matrix = np.abs(corr_matrix_real - corr_matrix_syn)

    """
    print("\n Correlation Difference Matrix\n")
    print(diff_matrix)
    """

    diff_matrix = diff_matrix.replace(np.diag(np.ones(diff_matrix.shape[0])), np.nan)

    # Count values below the threshold
    count_below_threshold = (diff_matrix < threshold).sum().sum()
    count_below_threshold = count_below_threshold / 2

    """print(f"\nNumber of values below {threshold}: {count_below_threshold}")"""

    number_of_relations = (len(num_features) * (len(num_features) - 1)) / 2

    """print(f"Number of relations (numerical features): {number_of_relations}")"""

    return round(count_below_threshold/ number_of_relations, 4)

"""
Chapter 2: Multivariate Relationship Analysis
    2.2 Normalized contingency tables comparison
"""
"""def normalized_contingency_tables(real_data, synthetic_data, attributes):
    results = {}
    for attr in attributes:
        table = pd.crosstab(real_data[attr], synthetic_data[attr], normalize='all')

        # Calculate the absolute deviation correctly
        expected = np.outer(table.sum(axis=0), table.sum(axis=1)) / table.sum().sum()
        absolute_deviation = np.sum(np.abs(table - expected))

        table_md = table.to_markdown(numalign='left', stralign='left')

        results[attr] = {
            "Contingency tables (Markdown)": table_md,
            "Absolute deviation": absolute_deviation
        }

    # Output the results
    for attr, values in results.items():
        print(f"\nAttribute: {attr}")
        print(values["Contingency tables (Markdown)"])
        print(f"Absolute deviation: {values['Absolute deviation']:.4f}")  # Formatting to 4 decimal places

    return results"""


"""
Chapter 3: DLA 
"""
def data_labelling_analysis(real_data, synthetic_data, num_features, cat_features):
    # Label real and synthetic data
    real_data["label"] = 0
    synthetic_data["label"] = 1

    # Combine both datasets to one
    combined_df = pd.concat([real_data, synthetic_data], axis=0)

    combined_df = pd.get_dummies(combined_df)

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









