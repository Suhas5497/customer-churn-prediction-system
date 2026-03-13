from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "telco_churn.csv"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports" / "figures"
TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"
TARGET_MAPPING = {"No": 0, "Yes": 1}


def ensure_directories() -> None:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "models").mkdir(parents=True, exist_ok=True)


def load_raw_data(data_path: Path | str = RAW_DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(data_path)


def clean_telco_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    cleaned = df.copy()
    initial_rows = len(cleaned)
    duplicate_rows = int(cleaned.duplicated().sum())

    cleaned = cleaned.drop_duplicates().copy()
    cleaned["TotalCharges"] = pd.to_numeric(cleaned["TotalCharges"], errors="coerce")
    invalid_total_charges = int(cleaned["TotalCharges"].isna().sum())

    cleaned = cleaned.dropna(subset=["TotalCharges"]).copy()
    cleaned = cleaned.drop(columns=[ID_COLUMN], errors="ignore")

    cleaned["SeniorCitizen"] = cleaned["SeniorCitizen"].astype(int)
    cleaned["tenure"] = cleaned["tenure"].astype(int)
    cleaned["MonthlyCharges"] = cleaned["MonthlyCharges"].astype(float)
    cleaned["TotalCharges"] = cleaned["TotalCharges"].astype(float)
    cleaned = cleaned.reset_index(drop=True)

    audit_summary = {
        "initial_rows": initial_rows,
        "duplicate_rows_removed": duplicate_rows,
        "invalid_total_charges_removed": invalid_total_charges,
        "rows_after_cleaning": len(cleaned),
    }
    return cleaned, audit_summary


def save_clean_dataset(output_path: Path | str = PROCESSED_DATA_DIR / "clean_telco_churn.csv") -> pd.DataFrame:
    ensure_directories()
    cleaned_df, _ = clean_telco_data(load_raw_data())
    cleaned_df.to_csv(output_path, index=False)
    return cleaned_df


def prepare_features_and_target(clean_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    modelling_df = clean_df.copy()
    y = modelling_df[TARGET_COLUMN].map(TARGET_MAPPING).astype(int)
    X = modelling_df.drop(columns=[TARGET_COLUMN])
    return X, y


def split_clean_dataset(
    clean_df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X, y = prepare_features_and_target(clean_df)
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def fit_preprocessing_artifacts(X_train: pd.DataFrame) -> dict[str, Any]:
    categorical_columns = X_train.select_dtypes(include="object").columns.tolist()
    numerical_columns = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()

    scaler = StandardScaler()
    scaler.fit(X_train[numerical_columns])

    encoded_train = pd.get_dummies(X_train, columns=categorical_columns, drop_first=True)

    default_input_values: dict[str, Any] = {}
    for column in X_train.columns:
        if column in categorical_columns:
            default_input_values[column] = X_train[column].mode().iloc[0]
        else:
            default_input_values[column] = float(X_train[column].median())

    category_levels = {
        column: sorted(X_train[column].dropna().astype(str).unique().tolist())
        for column in categorical_columns
    }

    return {
        "input_columns": X_train.columns.tolist(),
        "categorical_columns": categorical_columns,
        "numerical_columns": numerical_columns,
        "feature_columns": encoded_train.columns.tolist(),
        "default_input_values": default_input_values,
        "category_levels": category_levels,
        "scaler": scaler,
    }


def validate_input_frame(
    df: pd.DataFrame,
    preprocessing_artifacts: dict[str, Any],
    strict: bool = False,
) -> pd.DataFrame:
    expected_columns = preprocessing_artifacts["input_columns"]
    categorical_columns = preprocessing_artifacts["categorical_columns"]
    numerical_columns = preprocessing_artifacts["numerical_columns"]
    default_input_values = preprocessing_artifacts["default_input_values"]
    category_levels = preprocessing_artifacts["category_levels"]

    frame = df.copy()

    unknown_columns = sorted(set(frame.columns) - set(expected_columns))
    if unknown_columns:
        raise ValueError(f"Unknown input fields: {unknown_columns}")

    missing_columns = [column for column in expected_columns if column not in frame.columns]
    if missing_columns and strict:
        raise ValueError(f"Missing required input fields: {missing_columns}")

    for column in missing_columns:
        frame[column] = default_input_values[column]

    frame = frame[expected_columns]

    for column in categorical_columns:
        frame[column] = frame[column].fillna(default_input_values[column]).astype(str)
        invalid_values = sorted(set(frame[column]) - set(category_levels[column]))
        if invalid_values:
            raise ValueError(
                f"Invalid value(s) for '{column}': {invalid_values}. "
                f"Expected one of {category_levels[column]}."
            )

    for column in numerical_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if strict and frame[column].isna().any():
            raise ValueError(f"Column '{column}' contains missing or non-numeric values.")
        frame[column] = frame[column].fillna(default_input_values[column]).astype(float)

    return frame


def transform_features(
    X: pd.DataFrame,
    preprocessing_artifacts: dict[str, Any],
    strict: bool = False,
) -> pd.DataFrame:
    validated = validate_input_frame(X, preprocessing_artifacts, strict=strict)
    categorical_columns = preprocessing_artifacts["categorical_columns"]
    numerical_columns = preprocessing_artifacts["numerical_columns"]
    feature_columns = preprocessing_artifacts["feature_columns"]
    scaler: StandardScaler = preprocessing_artifacts["scaler"]

    encoded = pd.get_dummies(validated, columns=categorical_columns, drop_first=True)
    encoded = encoded.reindex(columns=feature_columns, fill_value=0)
    encoded[numerical_columns] = scaler.transform(validated[numerical_columns])

    return encoded.astype(float)


def create_preprocessed_datasets(
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    ensure_directories()
    clean_df = save_clean_dataset()
    X_train_raw, X_test_raw, y_train, y_test = split_clean_dataset(
        clean_df,
        test_size=test_size,
        random_state=random_state,
    )
    preprocessing_artifacts = fit_preprocessing_artifacts(X_train_raw)
    X_train = transform_features(X_train_raw, preprocessing_artifacts, strict=True)
    X_test = transform_features(X_test_raw, preprocessing_artifacts, strict=True)

    X_train.to_csv(PROCESSED_DATA_DIR / "X_train.csv", index=False)
    X_test.to_csv(PROCESSED_DATA_DIR / "X_test.csv", index=False)
    y_train.to_frame(name=TARGET_COLUMN).to_csv(PROCESSED_DATA_DIR / "y_train.csv", index=False)
    y_test.to_frame(name=TARGET_COLUMN).to_csv(PROCESSED_DATA_DIR / "y_test.csv", index=False)

    return {
        "clean_df": clean_df,
        "X_train_raw": X_train_raw.reset_index(drop=True),
        "X_test_raw": X_test_raw.reset_index(drop=True),
        "X_train": X_train.reset_index(drop=True),
        "X_test": X_test.reset_index(drop=True),
        "y_train": y_train.reset_index(drop=True),
        "y_test": y_test.reset_index(drop=True),
        "preprocessing_artifacts": preprocessing_artifacts,
    }


if __name__ == "__main__":
    datasets = create_preprocessed_datasets()
    print("Clean dataset shape:", datasets["clean_df"].shape)
    print("Training feature matrix shape:", datasets["X_train"].shape)
    print("Test feature matrix shape:", datasets["X_test"].shape)
