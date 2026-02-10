import pandas as pd
import numpy as np

def load_data(filepath):
    """
    Loads the dataset from the specified filepath.
    """
    data = pd.read_csv(filepath)
    return data

def preprocess_data(data):
    """
    Performs feature engineering and preprocessing on the data.
    """
    df = data.copy()

    # 1. Tenure Buckets
    tenure_bins = [0, 6, 12, 24, 48, 72, np.inf]
    tenure_labels = ["0-6", "6-12", "12-24", "24-48", "48-72", "72+"]
    df["tenure_bucket"] = pd.cut(
        df["tenure"],
        bins=tenure_bins,
        labels=tenure_labels,
        include_lowest=True,
        right=True
    )

    # 2. Charges per tenure
    # Note: Adding 1 to avoid division by zero if tenure is 0
    df["charges_per_tenure"] = df["MonthlyCharges"] / (df["tenure"] + 1)

    # 3. Family status
    # 'Yes' if either Partner or Dependents is 'Yes'
    df["family_status"] = (
        (df["Partner"].astype(str).str.strip().str.lower() == "yes") |
        (df["Dependents"].astype(str).str.strip().str.lower() == "yes")
    ).astype(int)

    # 4. Auto payment
    auto_methods = {"bank transfer (automatic)", "credit card (automatic)"}
    df["is_auto_payment"] = (
        df["PaymentMethod"].astype(str)
        .str.strip()
        .str.lower()
        .isin(auto_methods)
    ).astype(int)

    # 5. Binary Mapping (Yes/No to 1/0)
    binary_map_cols = ['Partner', 'Dependents', 'PaperlessBilling']
    for col in binary_map_cols:
        if col in df.columns:
            # Map 'Yes' to 1, 'No' to 0. defaults to NaN if mismatch (though data is clean)
            df[col] = df[col].map({'Yes': 1, 'No': 0})
    
    # Ensure SeniorCitizen is int
    if 'SeniorCitizen' in df.columns:
        df['SeniorCitizen'] = df['SeniorCitizen'].astype(int)

    return df

def get_features_and_target(df, target_col='Churn'):
    """
    Separates features and target.
    Maps target to 1/0 if it's categorical ('Yes'/'No').
    """
    if target_col in df.columns:
        y = df[target_col].map({'Yes': 1, 'No': 0})
        X = df.drop(columns=[target_col])
        return X, y
    else:
        # For inference where target might not be present
        return df, None
