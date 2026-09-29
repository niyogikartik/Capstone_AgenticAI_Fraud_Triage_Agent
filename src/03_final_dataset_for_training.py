def identify_and_drop_sparse_features(df_merged, threshold=0.7):
    # threshold = 0.7

    # Count the number of default values
    sparse_counts = (df_merged == -999).sum() + (df_merged == 'Unknown').sum()

    # Calculate the proportion of default values
    sparsity = sparse_counts / len(df_merged)

    # Identify columns to drop
    cols_to_drop = sparsity[sparsity > threshold].index.tolist()

    # Drop columns
    df_merged.drop(columns=cols_to_drop, inplace=True)

    print(f"Dropped {len(cols_to_drop)} sparse features.")
    print(f"New shape of df_merged: {df_merged.shape}")
    return df_merged, cols_to_drop

def remove_highly_correlated_features(df_merged, threshold=0.9):
    import numpy as np

    # Select numerical columns
    numeric_cols = df_merged.select_dtypes(include=['number']).columns

    # Compute correlation matrix
    corr_matrix = df_merged[numeric_cols].corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Find features with correlation greater than the threshold
    corr_col_to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    # Drop features
    df_merged.drop(columns=corr_col_to_drop, inplace=True)

    print(f"Dropped {len(corr_col_to_drop)} highly correlated features.")
    print(f"New shape of df_merged: {df_merged.shape}")
    return df_merged, corr_col_to_drop

def log_transform_amount(df_merged):
    import numpy as np
    df_merged['Log_TransactionAmt'] = np.log1p(df_merged['TransactionAmt'])
    (df_merged[['TransactionAmt', 'Log_TransactionAmt']].head())

    #visualize log transformed amount
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Plot raw TransactionAmt
    sns.histplot(df_merged['TransactionAmt'], bins=50, kde=True, ax=axes[0], color='blue')
    axes[0].set_title('Distribution of Raw TransactionAmt')
    axes[0].set_xlabel('Transaction Amount')
    axes[0].set_ylabel('Frequency')

    # Plot log-transformed TransactionAmt
    sns.histplot(df_merged['Log_TransactionAmt'], bins=50, kde=True, ax=axes[1], color='green')
    axes[1].set_title('Distribution of Log_TransactionAmt')
    axes[1].set_xlabel('Log(Transaction Amount + 1)')
    axes[1].set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()

    #Drop raw amount column
    df_merged.drop(columns=['TransactionAmt'], inplace=True)
    print(f"Dropped 'TransactionAmt'. New shape of df_merged: {df_merged.shape}")
    return df_merged

def extract_temporal_and_cyclical_features(df_merged):
    # Temporal features
    df_merged['Transaction_Hour'] = df_merged['TransactionDT_dt'].dt.hour
    df_merged['Transaction_DayOfWeek'] = df_merged['TransactionDT_dt'].dt.dayofweek
    df_merged['Is_Weekend'] = (df_merged['Transaction_DayOfWeek'] >= 5).astype(int)

    # Display the first few rows of the new temporal features   
    (df_merged[['TransactionDT_dt', 'Transaction_Hour', 'Transaction_DayOfWeek', 'Is_Weekend']].head())

    import numpy as np

    # Cyclical features for Transaction_Hour
    df_merged['Transaction_Hour_Sin'] = np.sin(2 * np.pi * df_merged['Transaction_Hour'] / 24)
    df_merged['Transaction_Hour_Cos'] = np.cos(2 * np.pi * df_merged['Transaction_Hour'] / 24)

    (df_merged[['Transaction_Hour', 'Transaction_Hour_Sin', 'Transaction_Hour_Cos']].head())

    #final dataset
    print(df_merged.columns.tolist())
    print(f"The final dataset has {(df_merged.shape)} features")
    return df_merged
