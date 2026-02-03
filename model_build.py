"""
model_built.py
--------------
Trains multiple ML models (LinearRegression, RandomForest, XGBoost)
on the preprocessed Pune property dataset.

 Loads preprocessed data
 Splits into train/test sets
 Trains multiple regression models
 Evaluates R² and RMSE
 Saves the best-performing model
 Also saves training-time feature column list (model_columns.pkl)
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor

def load_preprocessed_data(file_path: str):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f" File not found: {file_path}")
    print(f" Loading preprocessed dataset from {file_path} ...")

    df = pd.read_csv(file_path)
    print(f" Data loaded successfully with shape {df.shape}")

    if "price" not in df.columns:
        raise ValueError(" Target column 'price' not found in dataset.")

    X = df.drop(columns=["price"])
    y = df["price"]

    if X.isnull().sum().sum() > 0:
        print(" Warning: Missing values detected — filling with 0.")
        X = X.fillna(0)

    return X, y


def train_models(X_train, y_train):
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": XGBRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            objective="reg:squarederror",
            verbosity=0,
        ),
    }
    trained_models = {}
    print("\n Training models ...")

    for name, model in models.items():
        try:
            print(f"\n Training {name} ...")
            model.fit(X_train, y_train)
            trained_models[name] = model
            print(f" {name} trained successfully.")
        except Exception as e:
            print(f" {name} training failed: {e}")

    return trained_models

def evaluate_models(models, X_test, y_test):
    results = []
    for name, model in models.items():
        try:
            preds = model.predict(X_test)
            r2 = r2_score(y_test, preds)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            results.append((name, r2, rmse))
            print(f" {name} → R²: {r2:.4f} | RMSE: {rmse:.4f}")
        except Exception as e:
            print(f"❗ Evaluation failed for {name}: {e}")
    return results


def save_best_model(models, results, X):
    if not results:
        print(" No successful models to save.")
        return

    best_model_name, best_r2, best_rmse = sorted(results, key=lambda x: x[1], reverse=True)[0]
    best_model = models[best_model_name]

    model_filename = f"best_model_{best_model_name}.pkl"
    joblib.dump(best_model, model_filename)
    joblib.dump(X.columns.tolist(), "model_columns.pkl")

    print(f"\n Best model: {best_model_name}")
    print(f"   R²: {best_r2:.4f} | RMSE: {best_rmse:.4f}")
    print(f" Model saved as '{model_filename}'")
    print(f" Feature columns saved as 'model_columns.pkl'")


def main():
    data_file = "Pune_property_data_preprocessed.csv"

    try:
    
        X, y = load_preprocessed_data(data_file)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        models = train_models(X_train, y_train)

        results = evaluate_models(models, X_test, y_test)

        save_best_model(models, results, X)

        print("\n Model building and evaluation completed successfully!")

    except Exception as e:
        print(f"\n Model building failed: {e}")


if __name__ == "__main__":
    main()
