"""
data_preprocessing.py
---------------------
Final version — ensures absolutely no string values ('Unknown') remain.
Fully numeric, model-ready output.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler


def load_dataset(file_path: str) -> pd.DataFrame:
    print(f" Loading cleaned dataset from {file_path} ...")
    df = pd.read_csv(file_path)
    print(f" Loaded successfully with shape {df.shape}")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    print(" Handling missing values ...")
    for col in df.columns:
        if df[col].dtype in ["int64", "float64"]:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna("Unknown")
    return df


def reduce_high_cardinality(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    print(f" Reducing high-cardinality columns (keeping top {top_n} categories)...")
    for col in df.select_dtypes(include="object").columns:
        top_categories = df[col].value_counts().nlargest(top_n).index
        df[col] = df[col].where(df[col].isin(top_categories), other="Other")
    return df


def encode_columns(df: pd.DataFrame, ordinal_cols: list) -> pd.DataFrame:
    print(" Encoding ordinal columns ...")
    if ordinal_cols:
        oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        df[ordinal_cols] = oe.fit_transform(df[ordinal_cols])
        print(f" Encoded {len(ordinal_cols)} ordinal columns.")

    print(" One-hot encoding nominal columns ...")
    nominal_cols = [c for c in df.select_dtypes(include="object").columns if c not in ordinal_cols]

    if nominal_cols:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        encoded = ohe.fit_transform(df[nominal_cols])
        encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(nominal_cols))
        df = pd.concat([df.drop(columns=nominal_cols), encoded_df], axis=1)
        print(f" One-hot encoded {len(nominal_cols)} columns into {encoded_df.shape[1]} features.")
    else:
        print("ℹ No nominal columns found for one-hot encoding.")
    return df

def scale_numeric(df: pd.DataFrame) -> pd.DataFrame:
    print(" Scaling numeric columns ...")
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    print(f" Scaled {len(numeric_cols)} numeric columns.")
    return df

def main():
    input_file = "Pune_property_data_cleaned.csv"
    output_file = "Pune_property_data_preprocessed.csv"

    df = load_dataset(input_file)

    if "price" not in df.columns:
        raise ValueError(" Target column 'price' missing from dataset!")
    y = df["price"]
    X = df.drop(columns=["price"])

    X = handle_missing_values(X)
    X = reduce_high_cardinality(X)
    ordinal_cols = ["bhk_num"] if "bhk_num" in X.columns else []
    X = encode_columns(X, ordinal_cols)

    print(" Converting all remaining values to numeric ...")
    X = X.applymap(lambda x: np.nan if str(x).strip().lower() == "unknown" else x)
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    X = scale_numeric(X)

    y = pd.to_numeric(y, errors="coerce").fillna(0)
    df_preprocessed = pd.concat([X, y.rename("price")], axis=1)

    df_preprocessed.to_csv(output_file, index=False)
    print(f"\n Preprocessed dataset saved to {output_file}")
    print(f" Preprocessing completed successfully! Final shape: {df_preprocessed.shape}")


if __name__ == "__main__":
    main()
