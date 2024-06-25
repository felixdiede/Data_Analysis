import pandas as pd
import os
from fairmlhealth import report, measure
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import os
from glob import glob
import styleframe

os.chdir("/Users/felixdiederichs/PycharmProjects/Results_Analysis/.venv/Data/synthetic/obesity")

num_features = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"]

# Basisverzeichnis für die Datendateien
data_directory = "/Users/felixdiederichs/PycharmProjects/Results_Analysis/.venv/Data/synthetic/obesity"

# Ausgabeverzeichnis für die Reports
report_directory = "/Users/felixdiederichs/PycharmProjects/Results_Analysis/.venv/Reports/Fairness_Reports/obesity"

# Sicherstellen, dass das Ausgabeverzeichnis existiert
os.makedirs(report_directory, exist_ok=True)

# Alle CSV-Dateien im Datenverzeichnis finden
data_files = glob(os.path.join(data_directory, "*.csv"))

for file_path in data_files:
    # Dateinamen ohne Erweiterung extrahieren (für den Report-Namen)
    file_name = os.path.splitext(os.path.basename(file_path))[0]

    # Daten laden
    data = pd.read_csv(file_path)

    data = pd.get_dummies(data)

    scaler = MinMaxScaler()
    data[num_features] = scaler.fit_transform(data[num_features])

    # Features und Zielvariable trennen
    X = data.drop("NObeyesdad", axis=1)
    y = data["NObeyesdad"]

    # Daten aufteilen
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Modell trainieren (hier könnte man verschiedene Modelle ausprobieren)
    model = XGBClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Pfad für den Fairness-Report erstellen
    report_path = os.path.join(report_directory, f"{file_name}.xlsx")

    # Fairness-Report erstellen und speichern
    fairness_report = report.compare(
        test_data=X_test,
        targets=y_test,
        protected_attr=X_test["Gender"],
        models=model,
        skip_performance=True
    )
    data = fairness_report.data
    data.to_excel(report_path, index=False)

    print(f"Fairness-Report für '{file_name}' erstellt: {report_path}")

