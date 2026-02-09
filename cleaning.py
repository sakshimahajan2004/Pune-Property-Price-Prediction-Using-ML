"""
cleaning.py
------------
Custom cleaning script for Pune property dataset.
Performs numeric extraction, text standardization,
and removal of unnecessary columns.
"""

import pandas as pd
import re


def load_data(file_path: str) -> pd.DataFrame:
    print(f"Loading dataset from {file_path} ...")
    df = pd.read_csv(file_path)
    print(f"Data loaded successfully with shape {df.shape}")
    return df


def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
        print("Dropped 'Unnamed: 0' column.")
    return df


def clean_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    print("Cleaning numeric columns ...")

    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    df["area"] = (
        df["area"]
        .astype(str)
        .str.replace(",", "")
        .str.extract(r"(\d+\.?\d*)")[0]
        .astype(float)
    )

    df["pricepersquare"] = (
        df["pricepersquare"]
        .astype(str)
        .str.replace(",", "")
        .str.extract(r"(\d+\.?\d*)")[0]
        .astype(float)
    )

    print("Numeric columns cleaned.")
    return df

def extract_bhk(df: pd.DataFrame) -> pd.DataFrame:
    print("Extracting BHK numbers ...")
    df["bhk_num"] = (
        df["bhk"]
        .astype(str)
        .str.extract(r"(\d+)")
        .astype(float)
    )
    return df


def clean_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    print("Standardizing text columns ...")
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip().str.title()
    return df

def handle_missing_and_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = df.shape[0]
    df = df.drop_duplicates()
    after = df.shape[0]
    print(f"Removed {before - after} duplicate rows.")
    df = df.fillna("Unknown")
    return df


def save_cleaned_data(df: pd.DataFrame, output_file: str):
    df.to_csv(output_file, index=False)
    print(f" Cleaned dataset saved to {output_file}")

def main():
    input_file = "Pune_property_data.csv"
    output_file = "Pune_property_data_cleaned.csv"

    df = load_data(input_file)
    df = drop_unused_columns(df)
    df = clean_numeric_columns(df)
    df = extract_bhk(df)
    df = clean_text_columns(df)
    df = handle_missing_and_duplicates(df)
    save_cleaned_data(df, output_file)

    print("\n Cleaning completed successfully!")


if __name__ == "__main__":
    main()
