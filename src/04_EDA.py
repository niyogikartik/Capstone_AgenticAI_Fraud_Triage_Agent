def correlation_of_num_features_with_fraud(df_merged):
    numeric_cols = df_merged.select_dtypes(include=['number']).columns

    # Compute correlations with isFraud
    correlations = df_merged[numeric_cols].corr()['isFraud'].drop('isFraud')

    # Sort correlations
    correlations_sorted = correlations.sort_values(ascending=False)

    print("Top 10 positively correlated features with isFraud:")
    print(correlations_sorted.head(10))
    print("\nTop 10 negatively correlated features with isFraud:")
    print(correlations_sorted.tail(10))
    
    # Heatmap of correlations
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Convert Series to DataFrame for heatmap
    corr_df = correlations_sorted.to_frame(name='Correlation with isFraud')

    plt.figure(figsize=(10, 15))
    sns.heatmap(corr_df, annot=True, cmap='coolwarm', vmin=-0.2, vmax=0.2, center=0)
    plt.title('Correlation of Numerical Features with isFraud')
    plt.tight_layout()
    plt.show()


