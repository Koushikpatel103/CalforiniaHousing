import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("housing.csv")


cols = [
    'median_income',
    'total_rooms',
    'population',
    'households'
]

df_clean = df.copy()

for col in cols:

    Q1 = df_clean[col].quantile(0.25)
    Q3 = df_clean[col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df_clean = df_clean[
        (df_clean[col] >= lower) &
        (df_clean[col] <= upper)
    ]

# Feature Engineering
df_clean["rooms_per_household"] = (
    df_clean["total_rooms"] /
    df_clean["households"]
)

df_clean["bedrooms_per_room"] = (
    df_clean["total_bedrooms"] /
    df_clean["total_rooms"]
)

df_clean["population_per_household"] = (
    df_clean["population"] /
    df_clean["households"]
)

# Features and target
X = df_clean.drop(
    "median_house_value",
    axis=1
)

y = df_clean["median_house_value"]

# Missing values
num_cols = X.select_dtypes(
    include=['int64','float64']
).columns

imputer = SimpleImputer(
    strategy='median'
)

X[num_cols] = imputer.fit_transform(
    X[num_cols]
)

# Encoding
X = pd.get_dummies(
    X,
    drop_first=True
)

# Convert bool
bool_cols = X.select_dtypes(
    include='bool'
).columns

X[bool_cols] = X[
    bool_cols
].astype(int)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(
    X_train
)

# Train model
model = LinearRegression()

model.fit(
    X_train,
    y_train
)

# Save model and scaler
joblib.dump(
    model,
    "model.pkl"
)

joblib.dump(
    scaler,
    "scaler.pkl"
)

joblib.dump(
    X.columns,
    "columns.pkl"
)

print("Saved successfully")