"""
model_evaluate.py
-----------------
Loads a trained model and evaluates it on preprocessed test data.
Ensures feature alignment with the training-time columns saved during model building.
"""

import os
import joblib
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import numpy as np


# -----------------------------------------------------
# 1. Load preprocessed data
# -----------------------------------------------------
def load_preprocessed_data(file_path: str):
    """Load preprocessed data and separate features and target."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f" File not found: {file_path}")

    print(f" Loading preprocessed dataset from {file_path} ...")
    df = pd.read_csv(file_path)
    print(f" Data loaded successfully with shape {df.shape}")

    if "price" not in df.columns:
        raise ValueError(" Target column 'price' not found in dataset.")

    X = df.drop(columns=["price"])
    y = df["price"]

    return X, y


# -----------------------------------------------------
# 2. Align features with training columns
# -----------------------------------------------------
def align_features(X: pd.DataFrame, model_columns_path="model_columns.pkl"):
    """Ensure the evaluation dataset matches the model’s training features."""
    if not os.path.exists(model_columns_path):
        raise FileNotFoundError(" model_columns.pkl not found — retrain your model first.")

    print(" Aligning features with training columns ...")
    trained_cols = joblib.load(model_columns_path)

    # Add missing columns with zeros
    for col in trained_cols:
        if col not in X.columns:
            X[col] = 0

    # Drop extra columns not seen during training
    X = X[trained_cols]

    print(f" Features aligned: {X.shape[1]} columns match training.")
    return X


# -----------------------------------------------------
# 3. Evaluate model
# -----------------------------------------------------
def evaluate_model(model, X_test, y_test):
    """Evaluate a trained model on the test set."""
    print(" Evaluating model performance ...")

    predictions = model.predict(X_test)

    r2 = r2_score(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)

    print("\n Model Performance:")
    print(f"   R² Score  : {r2:.4f}")
    print(f"   RMSE      : {rmse:.4f}")
    print(f"   MAE       : {mae:.4f}")

    return r2, rmse, mae


# -----------------------------------------------------
# 4. Main evaluation workflow
# -----------------------------------------------------
def main():
    data_file = "Pune_property_data_preprocessed.csv"
    model_file = "best_model_RandomForest.pkl"  # You can change this if needed
    model_columns_file = "model_columns.pkl"

    try:
        # Step 1: Load model
        if not os.path.exists(model_file):
            raise FileNotFoundError(f" Model file not found: {model_file}")

        print(f" Loading trained model from {model_file} ...")
        model = joblib.load(model_file)
        print(" Model loaded successfully.")

        # Step 2: Load preprocessed dataset
        X, y = load_preprocessed_data(data_file)

        # Step 3: Align columns with training-time structure
        X = align_features(X, model_columns_file)

        # Step 4: Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Step 5: Evaluate model
        evaluate_model(model, X_test, y_test)

        print("\nEvaluation completed successfully!")

    except Exception as e:
        print(f"\n Model evaluation failed: {e}")


# -----------------------------------------------------
# Entry point
# -----------------------------------------------------
if __name__ == "__main__":
    main()
