import pandas as pd


def add_tenure_bucket(df, tenure_col="tenure", new_col="NEW_TENURE_YEAR"):
    """
    Create tenure bucket feature like:
    0-1 Year, 1-2 Year, ..., 5-6 Year
    """
    new_df = df.copy()

    new_df.loc[(new_df[tenure_col] >= 0) & (new_df[tenure_col] <= 12), new_col] = "0-1 Year"
    new_df.loc[(new_df[tenure_col] > 12) & (new_df[tenure_col] <= 24), new_col] = "1-2 Year"
    new_df.loc[(new_df[tenure_col] > 24) & (new_df[tenure_col] <= 36), new_col] = "2-3 Year"
    new_df.loc[(new_df[tenure_col] > 36) & (new_df[tenure_col] <= 48), new_col] = "3-4 Year"
    new_df.loc[(new_df[tenure_col] > 48) & (new_df[tenure_col] <= 60), new_col] = "4-5 Year"
    new_df.loc[(new_df[tenure_col] > 60) & (new_df[tenure_col] <= 72), new_col] = "5-6 Year"

    return new_df


def add_no_protection_flag(df):
    """
    NEW_noProt = no OnlineBackup, no DeviceProtection, no TechSupport
    (expects one-hot encoded *_Yes columns)
    """
    new_df = df.copy()

    required_cols = ["OnlineBackup_Yes", "DeviceProtection_Yes", "TechSupport_Yes"]
    for col in required_cols:
        if col not in new_df.columns:
            raise KeyError(f"Required column '{col}' not found for NEW_noProt.")

    new_df["NEW_noProt"] = (
        (new_df["OnlineBackup_Yes"] == 0) &
        (new_df["DeviceProtection_Yes"] == 0) &
        (new_df["TechSupport_Yes"] == 0)
    ).astype(int)

    return new_df


def add_engagement_flags(df):
    """
    NEW_Engaged: 1- or 2-year contracts
    NEW_Young_Not_Engaged: not engaged & not senior
    (expects Contract_One year, Contract_Two year, SeniorCitizen)
    """
    new_df = df.copy()

    needed = ["Contract_One year", "Contract_Two year", "SeniorCitizen"]
    for col in needed:
        if col not in new_df.columns:
            raise KeyError(f"Required column '{col}' not found for engagement flags.")

    new_df["NEW_Engaged"] = (
        (new_df["Contract_One year"] == 1) |
        (new_df["Contract_Two year"] == 1)
    ).astype(int)

    new_df["NEW_Young_Not_Engaged"] = (
        (new_df["NEW_Engaged"] == 0) &
        (new_df["SeniorCitizen"] == 0)
    ).astype(int)

    return new_df


def add_streaming_flag(df):
    """
    NEW_FLAG_ANY_STREAMING = 1 if StreamingTV_Yes == 1 or StreamingMovies_Yes == 1
    """
    new_df = df.copy()

    needed = ["StreamingTV_Yes", "StreamingMovies_Yes"]
    for col in needed:
        if col not in new_df.columns:
            raise KeyError(f"Required column '{col}' not found for streaming flag.")

    new_df["NEW_FLAG_ANY_STREAMING"] = (
        (new_df["StreamingTV_Yes"] == 1) |
        (new_df["StreamingMovies_Yes"] == 1)
    ).astype(int)

    return new_df


def add_auto_payment_flag(df):
    """
    NEW_FLAG_AutoPayment = 1 if Credit card (automatic) or Bank transfer (automatic)
    one-hot columns are 1.
    """
    new_df = df.copy()

    auto_cols = [
        "PaymentMethod_Credit card (automatic)",
        "PaymentMethod_Bank transfer (automatic)",
    ]
    auto_cols = [c for c in auto_cols if c in new_df.columns]

    if not auto_cols:
        raise KeyError("No automatic payment columns found for NEW_FLAG_AutoPayment.")

    new_df["NEW_FLAG_AutoPayment"] = (new_df[auto_cols].sum(axis=1) > 0).astype(int)

    return new_df


def add_service_agg_features(df):
    """
    NEW_TotalServices: count of *_Yes one-hot columns
    NEW_AVG_Service_Fee: MonthlyCharges / (NEW_TotalServices + 1)
    """
    new_df = df.copy()

    yes_cols = [c for c in new_df.columns if c.endswith("_Yes")]
    if not yes_cols:
        raise ValueError("No *_Yes columns found to compute NEW_TotalServices.")

    if "NEW_TotalServices" not in new_df.columns:
        new_df["NEW_TotalServices"] = new_df[yes_cols].sum(axis=1)

    if "MonthlyCharges" not in new_df.columns:
        raise KeyError("Column 'MonthlyCharges' is required for NEW_AVG_Service_Fee.")

    new_df["NEW_AVG_Service_Fee"] = new_df["MonthlyCharges"] / (
        new_df["NEW_TotalServices"] + 1
    )

    return new_df







# Full feature-engineering pipeline

def engineer_telco_features(df, output_path=r"..\data\extended_featured_data.csv", save=True):
    """
    Full feature-engineering pipeline for Telco dataset on an already
    preprocessed / encoded dataframe.

    Steps:
      - tenure buckets (NEW_TENURE_YEAR)
      - NEW_noProt
      - NEW_Engaged, NEW_Young_Not_Engaged
      - NEW_FLAG_ANY_STREAMING
      - NEW_FLAG_AutoPayment
      - NEW_TotalServices, NEW_AVG_Service_Fee
      - optionally save to CSV

    Returns
    -------
    new_features : pandas.DataFrame
        DataFrame with additional engineered features.
    """

    new_features = df.copy()

    # 1) tenure buckets
    new_features = add_tenure_bucket(new_features)

    # 2) protection flags
    new_features = add_no_protection_flag(new_features)

    # 3) engagement-related flags
    new_features = add_engagement_flags(new_features)

    # 4) streaming flag
    new_features = add_streaming_flag(new_features)

    # 5) automatic payment flag
    new_features = add_auto_payment_flag(new_features)

    # 6) aggregated service features
    new_features = add_service_agg_features(new_features)

    # 7) save to CSV if requested
    if save:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        new_features.to_csv(output_path, index=False)
        print(f"✅ Extended feature data saved to: {output_path}")

    return new_features
