import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# ==================================================
# BASE DIRECTORY
# ==================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR, "datasets", "diabetes.csv"
)  
# ⬆️ Rename your CSV to this OR change name here


MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)


# ==================================================
# LOAD DATASET
# ==================================================

df = pd.read_csv(DATA_PATH, na_values=["NA", "na", "null", "NULL", ""])

print("\n===== DATA PREVIEW =====")
print(df.head())

print("\nShape:", df.shape)

print("\nMissing values:")
print(df.isnull().sum())


# ==================================================
# TARGET COLUMN (THIS DATASET)
# ==================================================

# We will predict hypertension category
TARGET_COL = "Hypertension_NHLBI"


# ==================================================
# INPUT / OUTPUT
# ==================================================

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

print("\nInputs:", X.shape)
print("Target:", y.shape)


# ==================================================
# ENCODE TARGET (Normal=0, Others=1)
# ==================================================

y = y.replace({
    "Normal": 0,
    "Prehypertension": 1,
    "HTN-1": 1,
    "HTN-2": 1
})


# ==================================================
# COLUMN TYPES
# ==================================================

num_cols = X.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X.select_dtypes(include=["object", "bool"]).columns


# ==================================================
# HANDLE MISSING VALUES
# ==================================================

# Numeric
num_imputer = SimpleImputer(strategy="median")
X[num_cols] = num_imputer.fit_transform(X[num_cols])

# Categorical
cat_imputer = SimpleImputer(strategy="most_frequent")
X[cat_cols] = cat_imputer.fit_transform(X[cat_cols])


# ==================================================
# ONE-HOT ENCODING
# ==================================================

X = pd.get_dummies(X, columns=cat_cols, drop_first=True)


# ==================================================
# FEATURE SCALING
# ==================================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ==================================================
# TRAIN / TEST SPLIT
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==================================================
# ANN MODEL
# ==================================================

model = Sequential([

    Dense(128, activation="relu", input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.4),

    Dense(64, activation="relu"),
    BatchNormalization(),
    Dropout(0.3),

    Dense(32, activation="relu"),
    Dropout(0.2),

    Dense(1, activation="sigmoid")
])


model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# ==================================================
# CALLBACKS
# ==================================================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=15,
    restore_best_weights=True
)

lr_reduce = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=7,
    min_lr=1e-6
)


# ==================================================
# TRAIN
# ==================================================

print("\n===== TRAINING ANN =====")

history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop, lr_reduce],
    verbose=1
)


# ==================================================
# EVALUATION
# ==================================================

y_prob = model.predict(X_test)
y_pred = (y_prob > 0.5).astype(int)

acc = accuracy_score(y_test, y_pred)

print("\n===== PERFORMANCE =====")
print("Accuracy:", round(acc * 100, 2), "%")

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Hypertension Prediction")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# ==================================================
# SAVE FOR AI BOOT
# ==================================================

model.save(os.path.join(MODEL_DIR, "bp_ann_model.keras"))

joblib.dump(scaler, os.path.join(MODEL_DIR, "bp_scaler.pkl"))
joblib.dump(num_imputer, os.path.join(MODEL_DIR, "bp_num_imputer.pkl"))
joblib.dump(cat_imputer, os.path.join(MODEL_DIR, "bp_cat_imputer.pkl"))
joblib.dump(list(X.columns), os.path.join(MODEL_DIR, "bp_features.pkl"))


print("\n✅ SAVED FILES:")
print(" - bp_ann_model.keras")
print(" - bp_scaler.pkl")
print(" - bp_num_imputer.pkl")
print(" - bp_cat_imputer.pkl")
print(" - bp_features.pkl")


# ==================================================
# TERMINAL PREDICTION
# ==================================================

def predict_from_terminal():

    print("\n=================================")
    print(" BLOOD PRESSURE PREDICTION SYSTEM ")
    print("=================================\n")

    model = load_model(os.path.join(MODEL_DIR, "bp_ann_model.keras"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "bp_scaler.pkl"))
    num_imputer = joblib.load(os.path.join(MODEL_DIR, "bp_num_imputer.pkl"))
    cat_imputer = joblib.load(os.path.join(MODEL_DIR, "bp_cat_imputer.pkl"))
    features = joblib.load(os.path.join(MODEL_DIR, "bp_features.pkl"))

    user_data = {}

    print("Enter Patient Details:\n")

    for col in df.drop(columns=[TARGET_COL]).columns:

        while True:

            val = input(f"{col}: ")

            if val.strip() != "":
                user_data[col] = val
                break

            print("❌ Cannot be empty")


    user_df = pd.DataFrame([user_data])


    # Split cols
    user_df[num_cols] = num_imputer.transform(user_df[num_cols])
    user_df[cat_cols] = cat_imputer.transform(user_df[cat_cols])

    user_df = pd.get_dummies(user_df, columns=cat_cols, drop_first=True)


    # Align columns
    user_df = user_df.reindex(columns=features, fill_value=0)


    # Scale
    user_scaled = scaler.transform(user_df)


    # Predict
    prob = model.predict(user_scaled)[0][0]
    pred = 1 if prob >= 0.5 else 0


    print("\n=================================")

    if pred == 1:
        print("⚠️  HIGH RISK OF HYPERTENSION")
    else:
        print("✅ NORMAL BP")

    print("Confidence:", round(prob * 100, 2), "%")
    print("=================================\n")


# ==================================================
# RUN
# ==================================================

if __name__ == "__main__":
    predict_from_terminal()
