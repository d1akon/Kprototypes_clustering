import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def visualize_clusters(data, X_num_scaled, X_cat_encoded, labels):
    """
    Visualizes the clusters using PCA and TSNE.
    
    :param data: Original DataFrame with assigned clusters.
    :param X_num_scaled: Scaled numerical data.
    :param X_cat_encoded: Encoded categorical data.
    :param labels: Clusters assigned by the model.
    """
    #----- Combine numerical and categorical data
    X_combined = pd.concat([X_num_scaled, X_cat_encoded], axis=1)

    #----- PCA to reduce dimensions to 2D
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_combined)

    #----- TSNE to reduce dimensions to 2D
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    X_tsne = tsne.fit_transform(X_combined)

    #----- Add clusters and coordinates to DataFrame
    data['Cluster'] = labels
    data['PCA1'] = X_pca[:, 0]
    data['PCA2'] = X_pca[:, 1]
    data['TSNE1'] = X_tsne[:, 0]
    data['TSNE2'] = X_tsne[:, 1]

    #----- PCA Plot
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    sns.scatterplot(data=data, x='PCA1', y='PCA2', hue='Cluster', palette='tab10', s=50)
    plt.title('Clusters Visualized with PCA')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')

    #----- TSNE Plot
    plt.subplot(1, 2, 2)
    sns.scatterplot(data=data, x='TSNE1', y='TSNE2', hue='Cluster', palette='tab10', s=50)
    plt.title('Clusters Visualized with TSNE')
    plt.xlabel('TSNE Dimension 1')
    plt.ylabel('TSNE Dimension 2')

    plt.tight_layout()
    plt.show()


def analyze_cluster_characteristics(data, numeric_cols):
    """
    Analyzes the average numerical characteristics of each cluster.
    
    :param data: Original DataFrame with assigned clusters.
    :param numeric_cols: List of numerical columns.
    """
    #----- Group by cluster and calculate averages
    cluster_means = data.groupby('Cluster')[numeric_cols].mean()
    print("Averages per cluster:")
    print(cluster_means)

    #----- Visualization of features by cluster
    cluster_means.T.plot(kind='bar', figsize=(12, 6))
    plt.title("Average Numerical Features by Cluster")
    plt.ylabel("Average Value (scaled)")
    plt.xlabel("Features")
    plt.xticks(rotation=45)
    plt.legend(title="Cluster")
    plt.show()
