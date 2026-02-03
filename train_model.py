import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import os

# LOAD DATA
df = pd.read_csv("dataset/bmw.csv")

if "Unnamed: 0" in df.columns:
    df.drop(columns=["Unnamed: 0"], inplace=True)

# ENCODER
le_model = LabelEncoder()
le_trans = LabelEncoder()
le_fuel = LabelEncoder()

df["model"] = le_model.fit_transform(df["model"])
df["transmission"] = le_trans.fit_transform(df["transmission"])
df["fuelType"] = le_fuel.fit_transform(df["fuelType"])

FEATURES = [
    "model",
    "year",
    "transmission",
    "mileage",
    "fuelType",
    "tax",
    "mpg",
    "engineSize"
]

X = df[FEATURES]
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=300,
    random_state=42
)

model.fit(X_train, y_train)

os.makedirs("model", exist_ok=True)

joblib.dump(model, "model/model_bmw.pkl")
joblib.dump(le_model, "model/le_model.pkl")
joblib.dump(le_trans, "model/le_trans.pkl")
joblib.dump(le_fuel, "model/le_fuel.pkl")

print("✅ Model berhasil dilatih (8 fitur, aman)")
