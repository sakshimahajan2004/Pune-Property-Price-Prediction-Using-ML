"""
data_ingestion.py
-----------------
Handles data ingestion (loading + validation) for the Pune property dataset
and integrates the cleaning pipeline from cleaning.py.
"""

import os
import pandas as pd
from cleaning import (
    drop_unused_columns,
    clean_numeric_columns,
    extract_bhk,
    clean_text_columns,
    handle_missing_and_duplicates,
    save_cleaned_data
)


def validate_file_path(file_path: str) -> bool:
    """Check if the file exists and is a CSV."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    if not file_path.endswith(".csv"):
        raise ValueError(" Only CSV files are supported.")
    print(f" File path validated: {file_path}")
    return True


def load_raw_data(file_path: str) -> pd.DataFrame:
    """Load the raw dataset into a pandas DataFrame."""
    print(f" Loading data from {file_path} ...")
    df = pd.read_csv('Pune_property_data.csv')
    print(f" Data loaded successfully with shape {df.shape}")
    return df


def validate_data(df: pd.DataFrame, min_rows: int = 10) -> None:
    """Check for minimal data sanity (rows, columns)."""
    if df.empty:
        raise ValueError(" Loaded dataset is empty.")
    if df.shape[0] < min_rows:
        raise ValueError(f" Dataset has too few rows ({df.shape[0]}).")
    if df.shape[1] < 2:
        raise ValueError(" Dataset must have at least 2 columns.")
    print(f" Data validation passed: {df.shape[0]} rows, {df.shape[1]} columns.")


def run_cleaning_pipeline(df: pd.DataFrame, output_file: str) -> pd.DataFrame:
    """Run the cleaning steps sequentially from cleaning.py."""
    print("\n Running data cleaning pipeline ...")

    df = drop_unused_columns(df)
    df = clean_numeric_columns(df)
    df = extract_bhk(df)
    df = clean_text_columns(df)
    df = handle_missing_and_duplicates(df)

    save_cleaned_data(df, output_file)
    print(" Data cleaning completed successfully!")
    return df


def main():
    """Full data ingestion + cleaning pipeline."""
    input_file = "Pune_property_data.csv"
    output_file = "Pune_property_data_cleaned.csv"

    try:
        
        validate_file_path(input_file)
        df = load_raw_data(input_file)
        validate_data(df)

        df_cleaned = run_cleaning_pipeline(df, output_file)

        print(f"\n Full pipeline completed! Cleaned file saved at: {output_file}")
        print(f"Final shape: {df_cleaned.shape}")

    except Exception as e:
        print(f"\n Pipeline failed: {e}")


if __name__ == "__main__":
    main()
