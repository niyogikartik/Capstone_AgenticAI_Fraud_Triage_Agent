def train_catboost_model(
    enriched_features_refined,
    txn_labels,
    train_idx=None,
    val_idx=None,
    test_size=0.2,
    random_state=42,
):
    from catboost import CatBoostClassifier
    import numpy as np
    from sklearn.model_selection import train_test_split

    y = np.asarray(txn_labels)

    if train_idx is None or val_idx is None:
        indices = np.arange(len(y))
        train_idx, val_idx, y_train, y_val = train_test_split(
            indices,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )
    else:
        train_idx = np.asarray(train_idx)
        val_idx = np.asarray(val_idx)
        y_train = y[train_idx]
        y_val = y[val_idx]

    X_train_refined = enriched_features_refined[train_idx]
    X_val_refined = enriched_features_refined[val_idx]

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"Calculated scale_pos_weight: {scale_pos_weight:.4f}")

    clf_tuned = CatBoostClassifier(
        iterations=1000,
        depth=6,
        learning_rate=0.01,
        scale_pos_weight=scale_pos_weight,
        eval_metric='AUC',
        early_stopping_rounds=50,
        random_seed=42,
        verbose=False,
    )

    clf_tuned.fit(X_train_refined, y_train, eval_set=(X_val_refined, y_val))
    print("Tuned CatBoost model trained successfully.")
    print(f"Best iteration: {clf_tuned.get_best_iteration()}")
    print(f"Best validation AUC: {clf_tuned.get_best_score()['validation']['AUC']}")

    return clf_tuned, train_idx, val_idx, y_train, y_val

def evaluate_tuned_catboost_model(clf_tuned, X_val_refined, y_val):
    from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

    # Predict probabilities and labels using a custom decision threshold
    threshold = 0.86
    y_pred_proba_tuned = clf_tuned.predict_proba(X_val_refined)[:, 1]
    y_pred_tuned = (y_pred_proba_tuned >= threshold).astype(int)

    # Calculate metrics
    roc_auc_tuned = roc_auc_score(y_val, y_pred_proba_tuned)
    print(f"Tuned Model Validation ROC-AUC: {roc_auc_tuned:.4f}\n")

    print("Classification Report:")
    print(classification_report(y_val, y_pred_tuned))

    print("Confusion Matrix:")
    cm = confusion_matrix(y_val, y_pred_tuned)
    print(cm)
    return y_val, y_pred_proba_tuned, y_pred_tuned, roc_auc_tuned

def visualize_roc_curve(y_val, y_pred_proba_tuned, roc_auc_tuned):
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    # Compute ROC curve metrics
    fpr_tuned, tpr_tuned, _ = roc_curve(y_val, y_pred_proba_tuned)

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_tuned, tpr_tuned, color='darkorange', lw=2, label=f'Tuned Model ROC curve (area = {roc_auc_tuned:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic - Tuned Model')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

def get_feature_importance(clf_tuned, tabular_feature_names, output_dir=None):
    from pathlib import Path
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Generate refined GNN embedding names
    gnn_feature_names_refined = [f'GNN_Emb_{i}' for i in range(16)]

    # Combine feature names
    all_feature_names_refined = tabular_feature_names + gnn_feature_names_refined

    # Extract feature importances from the tuned model
    importances_tuned = clf_tuned.get_feature_importance()

    # Create a DataFrame and sort by importance
    feature_importance_tuned_df = pd.DataFrame({'Feature': all_feature_names_refined, 'Importance': importances_tuned})
    feature_importance_tuned_df = feature_importance_tuned_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        feature_importance_tuned_df.to_csv(output_dir / "feature_importance.csv", index=False)
        print(f"Saved feature importance to {output_dir / 'feature_importance.csv'}")

    # Visualize top 20 features
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', hue='Feature', data=feature_importance_tuned_df.head(20), palette='viridis', legend=False)
    plt.title('Top 20 Feature Importances (Tuned CatBoost)')
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()
    return feature_importance_tuned_df

def calculate_SHAP_values(clf_tuned, X_val_refined, y_val, y_pred_tuned=None, threshold=0.86,
                          feature_names=None, output_dir=None):
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import shap

    # Fall back to thresholding the model's probabilities when labels aren't supplied
    if y_pred_tuned is None:
        y_pred_tuned = (clf_tuned.predict_proba(X_val_refined)[:, 1] >= threshold).astype(int)

    # Initialize the explainer with the tuned model
    explainer_tuned = shap.TreeExplainer(clf_tuned)

    # Calculate SHAP values for the refined validation set
    shap_values_tuned = explainer_tuned(X_val_refined)

    print("SHAP Explainer initialized and SHAP values calculated successfully for the tuned model.")

    # Filter SHAP values by prediction category.
    #Filter the refined validation features and tuned SHAP values into True Positives (TP), False Positives (FP), True Negatives (TN), and False Negatives (FN) categories.
    mask_tp_tuned = (y_val == 1) & (y_pred_tuned == 1)
    mask_fp_tuned = (y_val == 0) & (y_pred_tuned == 1)
    mask_tn_tuned = (y_val == 0) & (y_pred_tuned == 0)
    mask_fn_tuned = (y_val == 1) & (y_pred_tuned == 0)

    X_val_refined_tp = X_val_refined[mask_tp_tuned]
    shap_values_tuned_tp = shap_values_tuned[mask_tp_tuned]

    X_val_refined_fp = X_val_refined[mask_fp_tuned]
    shap_values_tuned_fp = shap_values_tuned[mask_fp_tuned]

    X_val_refined_tn = X_val_refined[mask_tn_tuned]
    shap_values_tuned_tn = shap_values_tuned[mask_tn_tuned]

    X_val_refined_fn = X_val_refined[mask_fn_tuned]
    shap_values_tuned_fn = shap_values_tuned[mask_fn_tuned]

    print(f"Tuned TP Count: {len(X_val_refined_tp)}")
    print(f"Tuned FP Count: {len(X_val_refined_fp)}")
    print(f"Tuned TN Count: {len(X_val_refined_tn)}")
    print(f"Tuned FN Count: {len(X_val_refined_fn)}")

    if output_dir is not None:
        if feature_names is None:
            feature_names = [f'f{i}' for i in range(X_val_refined.shape[1])]

        parts = []
        for label, mask in (('TP', mask_tp_tuned), ('FP', mask_fp_tuned), ('FN', mask_fn_tuned)):
            part = pd.DataFrame(
                shap_values_tuned.values[mask], columns=[f'SHAP_{n}' for n in feature_names]
            )
            part.insert(0, 'Category', label)
            part.insert(1, 'Val_Index', np.flatnonzero(mask))
            parts.append(part)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        shap_path = output_dir / 'Validation_TP_FP_FN.csv'
        pd.concat(parts, ignore_index=True).to_csv(shap_path, index=False)
        print(f"Saved TP/FP/FN SHAP values to {shap_path}")

    return shap_values_tuned, shap_values_tuned_tp, shap_values_tuned_fp, shap_values_tuned_tn, shap_values_tuned_fn

def calcluate_tuned_threshold(y_pred_proba_tuned, y_val, threshold=0.86, output_dir="model_output"):
    from pathlib import Path
    import numpy as np
    import pandas as pd
    from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

    metrics_list_tuned = []
    thresholds = np.arange(0.01, 1.0, 0.01)

    for threshold in thresholds:
        preds = (y_pred_proba_tuned >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, preds).ravel()

        # Sklearn handles zero division via the zero_division parameter
        precision = precision_score(y_val, preds, zero_division=0)
        recall = recall_score(y_val, preds, zero_division=0)
        f1 = f1_score(y_val, preds, zero_division=0)

        # Handle zero division for custom ratios
        fp_tp_ratio = fp / tp if tp > 0 else np.nan
        fn_tn_ratio = fn / tn if tn > 0 else np.nan

        metrics_list_tuned.append({
            'Threshold': threshold,
            'TN': tn,
            'FP': fp,
            'FN': fn,
            'TP': tp,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'FP/TP Ratio': fp_tp_ratio,
            'FN/TN Ratio': fn_tn_ratio
        })

    tuned_threshold_metrics_df = pd.DataFrame(metrics_list_tuned)
    print(tuned_threshold_metrics_df.head())

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    threshold_path = output_dir / 'Threshold_metrics_v2.csv'
    tuned_threshold_metrics_df.to_csv(threshold_path, index=False)
    print(f'Tuned threshold metrics successfully exported to {threshold_path}')

    #Plot threshold
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(12, 18))

    # Plot 1: Counts
    axes[0].plot(tuned_threshold_metrics_df['Threshold'], tuned_threshold_metrics_df['TN'], label='TN', color='blue')
    axes[0].plot(tuned_threshold_metrics_df['Threshold'], tuned_threshold_metrics_df['FP'], label='FP', color='red')
    axes[0].plot(tuned_threshold_metrics_df['Threshold'], tuned_threshold_metrics_df['FN'], label='FN', color='orange')
    axes[0].plot(tuned_threshold_metrics_df['Threshold'], tuned_threshold_metrics_df['TP'], label='TP', color='green')
    axes[0].set_title('Confusion Matrix Counts vs. Threshold')
    axes[0].set_xlabel('Threshold')
    axes[0].set_ylabel('Count')
    axes[0].legend()
    axes[0].grid(True)

    # Plot 2: Metrics
    axes[1].plot(tuned_threshold_metrics_df['Threshold'], tuned_threshold_metrics_df['Precision'], label='Precision', color='blue')
    axes[1].plot(tuned_threshold_metrics_df['Threshold'], tuned_threshold_metrics_df['Recall'], label='Recall', color='green')
    axes[1].plot(tuned_threshold_metrics_df['Threshold'], tuned_threshold_metrics_df['F1-Score'], label='F1-Score', color='orange')
    axes[1].set_title('Precision, Recall, and F1-Score vs. Threshold')
    axes[1].set_xlabel('Threshold')
    axes[1].set_ylabel('Score')
    axes[1].legend()
    axes[1].grid(True)

    # Plot 3: Ratios
    axes[2].plot(tuned_threshold_metrics_df['Threshold'], tuned_threshold_metrics_df['FP/TP Ratio'], label='FP/TP Ratio', color='red')
    axes[2].plot(tuned_threshold_metrics_df['Threshold'], tuned_threshold_metrics_df['FN/TN Ratio'], label='FN/TN Ratio', color='purple')
    axes[2].set_title('FP/TP and FN/TN Ratios vs. Threshold')
    axes[2].set_xlabel('Threshold')
    axes[2].set_ylabel('Ratio (Log Scale)')
    axes[2].set_yscale('log')
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()

    return tuned_threshold_metrics_df
