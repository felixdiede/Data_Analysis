import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings('always')  # "error", "ignore", "always", "default", "module" or "once"


def trtr(real_data, target, num_features):
    real_data = pd.get_dummies(real_data)

    scaler = StandardScaler()
    real_data[num_features] = scaler.fit_transform(real_data[num_features])

    X = real_data.drop(target, axis=1)
    y = real_data[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

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

            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            results[clf.__class__.__name__] = [precision, recall, accuracy, f1]

        results_df = pd.DataFrame(results, index=["Precision", "Recall", "Accuracy", "F1"])

        # Flatten the DataFrame into a Series
        results_series = results_df.stack()

        # Round the values to two decimal places
        results_series = results_series.round(4)

        # Return the Series of results
        return results_series




def tstr(real_data, synthetic_data, target, num_features):
    real_data = pd.get_dummies(real_data)
    synthetic_data = pd.get_dummies(synthetic_data)

    scaler = StandardScaler()
    real_data[num_features] = scaler.fit_transform(real_data[num_features])
    synthetic_data[num_features] = scaler.transform(synthetic_data[num_features])

    common_columns = synthetic_data.columns.intersection(real_data.columns)

    synthetic_data = synthetic_data[common_columns]
    real_data = real_data[common_columns]

    Xs = synthetic_data.drop(target, axis=1)
    ys = synthetic_data[target]

    Xr = real_data.drop(target, axis=1)
    yr = real_data[target]

    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(Xs, ys, test_size=0.2, random_state=0)

    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(Xr, yr, test_size=0.2, random_state=0)

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
            clf.fit(X_train_s, y_train_s)
            y_pred = clf.predict(X_test_r)

            precision = precision_score(y_test_r, y_pred)
            recall = recall_score(y_test_r, y_pred)
            accuracy = accuracy_score(y_test_r, y_pred)
            f1 = f1_score(y_test_r, y_pred)

            results[clf.__class__.__name__] = [precision, recall, accuracy, f1]

        results_df = pd.DataFrame(results, index=["Precision", "Recall", "Accuracy", "F1"])

        # Flatten the DataFrame into a Series
        results_series = results_df.stack()

        # Round the values to two decimal places
        results_series = results_series.round(4)

        # Return the Series of results
        return results_series



