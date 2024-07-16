import pandas as pd
import os
from fairlearn.metrics import demographic_parity_difference
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


cat_features = ["Gender", "family_history_with_overweight", "FAVC", "CAEC", "SMOKE", "SCC", "CALC", "MTRANS", "NObeyesdad"]
num_features = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"]

os.chdir("/Users/felixdiederichs/PycharmProjects/Data_Analysis/.venv/Data/synthetic/obesity/obesity_2/")

all_data = pd.DataFrame()
dataframes = {}

def custom_sort(file_name):
    if file_name.endswith(".csv"):
        file_name = file_name[:-4]

    parts = file_name.split("_")
    model = parts[1]

    # Check for lambda value presence before accessing parts[2] and parts[3]
    if len(parts) >= 4:  # Lambda value present
        epoch = int(parts[2])
        lambda_value = parts[3]
    else:  # No lambda value
        epoch = int(parts[2])
        lambda_value = ""  # Default empty string if no lambda

    model_priority = {"tabfairgan": 1, "distcorrgan": 2, "multifairgan": 3, "decaf": 4, "tvae": 5, "ctgan": 6}
    lambda_priority = {"02": 1, "04": 2, "06": 3, "08": 4, "1": 5, "15": 6, "2": 7, "5": 8}

    return model_priority.get(model, 99), epoch, lambda_priority.get(lambda_value, 99)


file_names = os.listdir()

sorted_files = sorted(file_names, key=custom_sort)


for file_names in sorted_files:
        file_path = os.path.join(file_names)
        dataframes[file_names] = pd.read_csv(file_path)

        synthetic_data = dataframes[file_names]

        X = synthetic_data.drop("NObeyesdad", axis=1)
        y = synthetic_data["NObeyesdad"]

        X = pd.get_dummies(X)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = DecisionTreeClassifier(random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_true = y_test

        spd = demographic_parity_difference(y_true, y_pred, sensitive_features=X_test["Gender"])

        print(spd)

        # results = pd.DataFrame(results, columns = [file_names])

        # all_data = pd.concat([all_data, results], axis=1)










