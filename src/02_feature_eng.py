def create_velocity_features(df_merged):
    import pandas as pd
    import numpy as np

    # Ensure TransactionDT_dt exists
    if 'TransactionDT_dt' not in df_merged.columns:
        df_merged['TransactionDT_dt'] = pd.to_datetime('2017-12-01') + pd.to_timedelta(df_merged['TransactionDT'], unit='s')

    # Keep track of original index to restore order later if needed, and to assign safely
    df_merged = df_merged.sort_values('TransactionDT_dt').reset_index(drop=True)

    # Compute rolling features
    # Set datetime index for rolling
    df_tmp = df_merged.set_index('TransactionDT_dt')

    # Group by
    grouped = df_tmp.groupby(['PaymentCard_Bank_ID', 'Merchant_Proxy_ID'])

    # Calculate rolling windows
    # 15 min
    roll_15m = grouped.rolling('15min')
    df_merged['Velocity_Count_15m'] = roll_15m['TransactionID'].count().reset_index(level=[0,1], drop=True).values
    df_merged['Velocity_Amt_15m'] = roll_15m['TransactionAmt'].sum().reset_index(level=[0,1], drop=True).values

    # 1 Hour
    roll_1h = grouped.rolling('1h')
    df_merged['Velocity_Count_1h'] = roll_1h['TransactionID'].count().reset_index(level=[0,1], drop=True).values
    df_merged['Velocity_Amt_1h'] = roll_1h['TransactionAmt'].sum().reset_index(level=[0,1], drop=True).values

    # 24 Hours
    roll_24h = grouped.rolling('24h')
    df_merged['Velocity_Count_24h'] = roll_24h['TransactionID'].count().reset_index(level=[0,1], drop=True).values
    df_merged['Velocity_Amt_24h'] = roll_24h['TransactionAmt'].sum().reset_index(level=[0,1], drop=True).values

    print("Rolling velocity features engineered successfully.")
    print(df_merged[['TransactionDT_dt', 'PaymentCard_Bank_ID', 'Merchant_Proxy_ID', 'Velocity_Count_15m', 'Velocity_Amt_15m', 'Velocity_Count_1h', 'Velocity_Amt_1h']].head(10))
    return df_merged

def handle_missing_values(df_merged):
    numeric_cols = df_merged.select_dtypes(include=['number']).columns
    categorical_cols = df_merged.select_dtypes(exclude=['number', 'datetime']).columns

    df_merged[numeric_cols] = df_merged[numeric_cols].fillna(-999)
    df_merged[categorical_cols] = df_merged[categorical_cols].fillna('Unknown')

    print(f"Max missing values after imputation: {df_merged.isnull().sum().max()}")
    return df_merged
