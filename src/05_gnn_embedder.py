import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Classifier(torch.nn.Module):
    """Maps transaction node embeddings to a single fraud logit."""
    def __init__(self, in_channels):
        super().__init__()
        self.lin = torch.nn.Linear(in_channels, 1)

    def forward(self, x):
        return self.lin(x)


def generate_graph_data(df_merged, target_col='isFraud', val_size=0.2, random_state=42):
    import numpy as np
    import pandas as pd
    from torch_geometric.data import HeteroData

    # 1. Mappings
    txn_mapping = {tid: i for i, tid in enumerate(df_merged['TransactionID'])}
    unique_cards = df_merged['PaymentCard_Bank_ID'].unique()
    card_mapping = {card_id: i for i, card_id in enumerate(unique_cards)}
    unique_merchants = df_merged['Merchant_Proxy_ID'].unique()
    merchant_mapping = {merch_id: i for i, merch_id in enumerate(unique_merchants)}

    # 2. Extract Features
    cols_to_drop = ['TransactionID', 'TransactionDT_dt', 'PaymentCard_Bank_ID', 'Merchant_Proxy_ID', 'ProductCD',
                    'PaymentCard_Billing_Country', 'PaymentCard_Type', 'PaymentCard_Category', 'PaymentCard_Network', 'addr1', 'addr2', 'P_emaildomain', 'Match_M1', 'Match_M2', 'Match_M3', 'Match_M4', 'Match_M5', 'Match_M6', 'Match_M7', 'Match_M8', 'Match_M9']

    cols_to_drop_actual = [col for col in cols_to_drop if col in df_merged.columns]
    txn_features_df = df_merged.drop(columns=cols_to_drop_actual)

    categorical_features = txn_features_df.select_dtypes(exclude=[np.number]).columns
    if len(categorical_features) > 0:
        txn_features_df = pd.get_dummies(txn_features_df, columns=categorical_features, drop_first=True)

    # Ensure feature columns align perfectly with train (in case of missing dummies)
    expected_cols = [
        'PaymentCard_Dummy1', 'dist1', 'Count_C1', 'Count_C3', 'Count_C5',
        'Days_Since_D1', 'Days_Since_D2', 'Days_Since_D4', 'Days_Since_D5',
        'Days_Since_D10', 'Days_Since_D11', 'Vesta_Feature_V95', 'Vesta_Feature_V98',
        'Vesta_Feature_V129', 'Vesta_Feature_V130', 'Vesta_Feature_V135',
        'Vesta_Feature_V281', 'Vesta_Feature_V284', 'Vesta_Feature_V291',
        'Vesta_Feature_V313', 'Vesta_Feature_V319', 'Velocity_Count_15m',
        'Velocity_Amt_15m', 'Velocity_Count_24h', 'Log_TransactionAmt',
        'Transaction_Hour', 'Transaction_DayOfWeek', 'Is_Weekend',
        'Transaction_Hour_Sin', 'Transaction_Hour_Cos'
    ]
    for col in expected_cols:
        if col not in txn_features_df.columns:
            txn_features_df[col] = 0
    txn_features_df = txn_features_df[expected_cols]

    txn_features = txn_features_df.values

    # 3. Initialize HeteroData
    graph_data = HeteroData()

    # 4. Set Transaction Features
    graph_data['transaction'].x = torch.tensor(txn_features, dtype=torch.float)

    # 5. Set Number of Nodes for Card and Merchant
    graph_data['card'].num_nodes = len(card_mapping)
    graph_data['merchant'].num_nodes = len(merchant_mapping)

    # 6. Create Edge Indices
    txn_indices = df_merged['TransactionID'].map(txn_mapping).values
    card_indices = df_merged['PaymentCard_Bank_ID'].map(card_mapping).values
    merchant_indices = df_merged['Merchant_Proxy_ID'].map(merchant_mapping).values

    edge_index_txn_card = torch.tensor(np.array([txn_indices, card_indices]), dtype=torch.long)
    graph_data['transaction', 'used', 'card'].edge_index = edge_index_txn_card

    edge_index_txn_merchant = torch.tensor(np.array([txn_indices, merchant_indices]), dtype=torch.long)
    graph_data['transaction', 'at', 'merchant'].edge_index = edge_index_txn_merchant

    # 7. Labels and train/val mask (needed for supervised GNN training)
    if target_col in df_merged.columns:
        from sklearn.model_selection import train_test_split

        labels = df_merged[target_col].values
        graph_data['transaction'].y = torch.tensor(labels, dtype=torch.long)

        indices = np.arange(len(labels))
        train_idx, val_idx = train_test_split(
            indices, test_size=val_size, random_state=random_state, stratify=labels
        )
        train_mask = torch.zeros(len(labels), dtype=torch.bool)
        val_mask = torch.zeros(len(labels), dtype=torch.bool)
        train_mask[train_idx] = True
        val_mask[val_idx] = True
        graph_data['transaction'].train_mask = train_mask
        graph_data['transaction'].val_mask = val_mask

    print(graph_data)


    return graph_data, txn_features


def train_gnn_model(graph_data, txn_features, epochs=10):
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn.functional as F
    from torch_geometric.nn import SAGEConv, to_hetero

    # Define a deeper and wider GNN model
    class DeeperWiderGNN(torch.nn.Module):
        def __init__(self, hidden_channels, out_channels):
            super().__init__()
            self.conv1 = SAGEConv((-1, -1), hidden_channels * 2)
            self.conv2 = SAGEConv((-1, -1), hidden_channels)
            self.conv3 = SAGEConv((-1, -1), out_channels)

        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index)
            x = x.relu()
            x = self.conv2(x, edge_index)
            x = x.relu()
            x = self.conv3(x, edge_index)
            return x

    hidden_channels = 32
    out_channels = 16

    base_model_refined = DeeperWiderGNN(hidden_channels, out_channels)
    model_refined = to_hetero(base_model_refined, graph_data.metadata(), aggr='sum').to(device)

    classifier_refined = Classifier(out_channels).to(device)
    optimizer_refined = torch.optim.Adam(list(model_refined.parameters()) + list(classifier_refined.parameters()), lr=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()

    model_refined.train()
    classifier_refined.train()
    for epoch in range(epochs):
        optimizer_refined.zero_grad()
        out = model_refined(graph_data.x_dict, graph_data.edge_index_dict)
        txn_emb = out['transaction']
        logits = classifier_refined(txn_emb).squeeze()

        loss = criterion(logits[graph_data['transaction'].train_mask], graph_data['transaction'].y[graph_data['transaction'].train_mask].float())
        loss.backward()
        optimizer_refined.step()

        if (epoch+1) % 2 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')

    print("Refined GNN training completed.")
    model_refined.eval()
    with torch.no_grad():
        out_refined = model_refined(graph_data.x_dict, graph_data.edge_index_dict)
        gnn_embeddings_refined = out_refined['transaction'].cpu().numpy()

    enriched_features_refined = np.hstack((txn_features, gnn_embeddings_refined))
    print(f"Original features shape: {txn_features.shape}")
    print(f"Refined GNN embeddings shape: {gnn_embeddings_refined.shape}")
    print(f"Refined enriched features shape: {enriched_features_refined.shape}")

    return enriched_features_refined


