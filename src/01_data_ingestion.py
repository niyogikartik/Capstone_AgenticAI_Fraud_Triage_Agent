import pandas as pd
import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

def load_and_merge_train_data():
    train_ID_file_path = config['data']['train_ID_file_path']
    train_transaction_file_path = config['data']['train_transaction_file_path']
    nrows = config['data'].get('nrows')
    df_train_ID = pd.read_csv(train_ID_file_path, nrows=nrows)
    df_train_transaction = pd.read_csv(train_transaction_file_path, nrows=nrows)
    df_train_merged = pd.merge(df_train_transaction, df_train_ID, on='TransactionID', how='left')
    print(f"Merged dataset shape: {df_train_merged.shape}")
    df_train_merged.head()
    return df_train_merged

def load_and_merge_test_data():
    test_ID_file_path = config['data']['test_ID_file_path']
    test_transaction_file_path = config['data']['test_transaction_file_path']
    nrows = config['data'].get('nrows')
    df_test_ID = pd.read_csv(test_ID_file_path, nrows=nrows)
    df_test_transaction = pd.read_csv(test_transaction_file_path, nrows=nrows)
    df_test_merged = pd.merge(df_test_transaction, df_test_ID, on='TransactionID', how='left')
    print(f"Merged dataset shape: {df_test_merged.shape}")
    df_test_merged.head()
    return df_test_merged

def rename_anonymized_features(df_merged):
    rename_dict = {
            'card1': 'PaymentCard_Bank_ID',
            'card2': 'PaymentCard_Billing_Country',
            'card3': 'PaymentCard_Dummy1',
            'card4': 'PaymentCard_Type',
            'card5': 'PaymentCard_Category',
            'card6': 'PaymentCard_Network'
    }

    # Rename card columns
    df_merged.rename(columns=rename_dict, inplace=True)

    # Prefix C, D, M, V columns
    new_cols = {}
    for col in df_merged.columns:
        if col.startswith('C') and col[1:].isdigit():
            new_cols[col] = 'Count_' + col
        elif col.startswith('D') and col[1:].isdigit():
            new_cols[col] = 'Days_Since_' + col
        elif col.startswith('M') and col[1:].isdigit():
            new_cols[col] = 'Match_' + col
        elif col.startswith('V') and col[1:].isdigit():
            new_cols[col] = 'Vesta_Feature_' + col

    df_merged.rename(columns=new_cols, inplace=True)
    print("Columns renamed successfully.")

    #Create merchant proxy features
    cols_to_combine = ['ProductCD', 'addr1', 'addr2']
    for col in cols_to_combine:
        df_merged[col] = df_merged[col].fillna('Unknown').astype(str)

    df_merged['Merchant_Proxy_ID'] = df_merged['ProductCD'] + '_' + df_merged['addr1'] + '_' + df_merged['addr2']
    return df_merged