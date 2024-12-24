import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score, adjusted_rand_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns
from src.model.kprototypes import KPrototypesCustom



def find_best_k(X_num_scaled, X_cat_encoded, max_clusters=10, gamma=1.0, penalty_alpha=0.1):
    """
    Finds the best value of k using normalized Silhouette Score, Davies-Bouldin Index, 
    and a penalty for high k values.
    :param X_num_scaled: Scaled numerical data.
    :param X_cat_encoded: Encoded categorical data.
    :param max_clusters: Maximum number of clusters to test.
    :param gamma: Weight for categorical distance.
    :param penalty_alpha: Penalty factor for high k values.
    :return: Dictionary with k values, Silhouette Scores, Davies-Bouldin Index, and combined metric.
    """
    silhouette_scores = []
    davies_bouldin_scores = []
    inertia = []
    k_values = list(range(2, max_clusters + 1))

    for k in k_values:
        print(f"Fitting for k = {k}...")

        # Train the model with the current k value
        model = KPrototypesCustom(n_clusters=k, gamma=gamma)
        model.fit(X_num_scaled.values, X_cat_encoded.values)

        # Calculate inertia
        total_inertia = 0
        for cluster in range(k):
            cluster_points = np.where(model.labels_ == cluster)[0]
            cluster_inertia = 0
            for point in cluster_points:
                cluster_inertia += np.sum(
                    (X_num_scaled.iloc[point] - model.centroids[0][cluster]) ** 2
                )
                cluster_inertia += gamma * np.sum(
                    X_cat_encoded.iloc[point] != model.centroids[1][cluster]
                )
            total_inertia += cluster_inertia
        inertia.append(total_inertia)

        # Calculate Silhouette Score
        labels = model.predict(X_num_scaled.values, X_cat_encoded.values)
        silhouette = silhouette_score(
            np.hstack((X_num_scaled, X_cat_encoded)), labels, metric='euclidean'
        )
        silhouette_scores.append(silhouette)

        # Calculate Davies-Bouldin Index
        dbi = davies_bouldin_score(np.hstack((X_num_scaled, X_cat_encoded)), labels)
        davies_bouldin_scores.append(dbi)

        print(f"k={k}: Inertia={total_inertia:.2f}, Silhouette Score={silhouette:.4f}, DBI={dbi:.4f}")

    # Normalize metrics
    silhouette_min, silhouette_max = min(silhouette_scores), max(silhouette_scores)
    normalized_silhouette_scores = [
        (s - silhouette_min) / (silhouette_max - silhouette_min) for s in silhouette_scores
    ]

    dbi_min, dbi_max = min(davies_bouldin_scores), max(davies_bouldin_scores)
    normalized_davies_bouldin_scores = [
        (dbi - dbi_min) / (dbi_max - dbi_min) for dbi in davies_bouldin_scores
    ]

    # Combined metric with normalized scores
    combined_metric = [
        normalized_silhouette_scores[i] - normalized_davies_bouldin_scores[i] - penalty_alpha * k_values[i]
        for i in range(len(k_values))
    ]
    best_k_combined = k_values[np.argmax(combined_metric)]

    # Plot the results
    plt.figure(figsize=(15, 5))

    # Silhouette Score plot
    plt.subplot(1, 3, 1)
    plt.plot(k_values, silhouette_scores, marker='o', linestyle='--')
    plt.title('Silhouette Score vs Number of Clusters')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')

    # Davies-Bouldin Index plot
    plt.subplot(1, 3, 2)
    plt.plot(k_values, davies_bouldin_scores, marker='o', linestyle='--', color='orange')
    plt.title('Davies-Bouldin Index vs Number of Clusters')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('DBI (lower is better)')

    # Combined Metric plot
    plt.subplot(1, 3, 3)
    plt.plot(k_values, combined_metric, marker='o', linestyle='--', color='green')
    plt.title('Combined Metric vs Number of Clusters')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Combined Metric')

    plt.tight_layout()
    plt.show()

    return {
        'k_values': k_values,
        'silhouette_scores': silhouette_scores,
        'davies_bouldin_scores': davies_bouldin_scores,
        'inertia': inertia,
        'combined_metric': combined_metric,
        'best_k_combined': best_k_combined,
    }




def evaluate_silhouette(X_num_scaled, X_cat_encoded, labels):
    """
    Calculates the Silhouette Score to evaluate the clusters.
    :param X_num_scaled: Scaled numerical data.
    :param X_cat_encoded: Encoded categorical data.
    :param labels: Cluster labels assigned.
    :return: Silhouette Score.
    """
    X_combined = pd.concat([X_num_scaled, X_cat_encoded], axis=1)
    score = silhouette_score(X_combined, labels, metric='euclidean')
    print(f"Silhouette Score: {score:.4f}")
    return score


def evaluate_inertia(model):
    """
    Returns the inertia of the model after fitting.
    :param model: Fitted K-Prototypes model.
    :return: Model inertia.
    """
    if model.centroids is None:
        raise ValueError("The model is not trained.")
    inertia = model.centroids[0].shape[0]  #----- Example of inertia
    print(f"Model inertia: {inertia}")
    return inertia


def evaluate_ari(true_labels, predicted_labels):
    """
    Calculates the ARI to evaluate the clusters.
    :param true_labels: True labels.
    :param predicted_labels: Labels assigned by the model.
    :return: Adjusted Rand Index.
    """
    score = adjusted_rand_score(true_labels, predicted_labels)
    print(f"Adjusted Rand Index: {score:.4f}")
    return score


def visualize_boxplot(data, numeric_cols):
    """
    Generates a boxplot to analyze how numerical features vary across clusters.
    :param data: DataFrame with data and cluster column.
    :param numeric_cols: List of numerical columns.
    """
    for col in numeric_cols:
        plt.figure(figsize=(8, 4))
        sns.boxplot(data=data, x='Cluster', y=col, palette='Set3')
        plt.title(f'Distribution of {col} by Cluster')
        plt.show()


def elbow_method(X_num_scaled, X_cat_encoded, max_clusters=10, gamma=1.0):
    """
    Uses the Elbow Method to find the optimal number of clusters.
    :param X_num_scaled: Scaled numerical data.
    :param X_cat_encoded: Encoded categorical data.
    :param max_clusters: Maximum number of clusters to test.
    :param gamma: Weight for categorical distance.
    """
    from projects.projects_github.Kprototypes_clustering.model.kprototypes import KPrototypesCustom

    inertia = []
    for k in range(2, max_clusters + 1):
        print(f"Calculating for k = {k}...")
        model = KPrototypesCustom(n_clusters=k, gamma=gamma)
        model.fit(X_num_scaled.values, X_cat_encoded.values)
        inertia.append(model.centroids[0].shape[0])  #----- Example of inertia

    #----- Plot inertia
    plt.figure(figsize=(8, 4))
    plt.plot(range(2, max_clusters + 1), inertia, marker='o', linestyle='--')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method')
    plt.show()
