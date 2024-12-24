import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def visualize_clusters(data, X_num_scaled, X_cat_encoded, labels):
    """
    Visualiza los clusters utilizando PCA y TSNE.
    
    :param data: DataFrame original con los clusters asignados.
    :param X_num_scaled: Datos numéricos escalados.
    :param X_cat_encoded: Datos categóricos codificados.
    :param labels: Clusters asignados por el modelo.
    """
    # Combinar datos numéricos y categóricos
    X_combined = pd.concat([X_num_scaled, X_cat_encoded], axis=1)

    # PCA para reducir dimensiones a 2D
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_combined)

    # TSNE para reducir dimensiones a 2D
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    X_tsne = tsne.fit_transform(X_combined)

    # Agregar clusters y coordenadas a un DataFrame
    data['Cluster'] = labels
    data['PCA1'] = X_pca[:, 0]
    data['PCA2'] = X_pca[:, 1]
    data['TSNE1'] = X_tsne[:, 0]
    data['TSNE2'] = X_tsne[:, 1]

    # Gráficos PCA
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    sns.scatterplot(data=data, x='PCA1', y='PCA2', hue='Cluster', palette='tab10', s=50)
    plt.title('Clusters visualizados con PCA')
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')

    # Gráficos TSNE
    plt.subplot(1, 2, 2)
    sns.scatterplot(data=data, x='TSNE1', y='TSNE2', hue='Cluster', palette='tab10', s=50)
    plt.title('Clusters visualizados con TSNE')
    plt.xlabel('Dimensión TSNE 1')
    plt.ylabel('Dimensión TSNE 2')

    plt.tight_layout()
    plt.show()


def analyze_cluster_characteristics(data, numeric_cols):
    """
    Analiza las características numéricas promedio de cada cluster.
    
    :param data: DataFrame original con los clusters asignados.
    :param numeric_cols: Lista de columnas numéricas.
    """
    # Agrupar por cluster y calcular promedios
    cluster_means = data.groupby('Cluster')[numeric_cols].mean()
    print("Promedios por cluster:")
    print(cluster_means)

    # Visualización de características por cluster
    cluster_means.T.plot(kind='bar', figsize=(12, 6))
    plt.title("Promedio de características numéricas por cluster")
    plt.ylabel("Valor promedio (escalado)")
    plt.xlabel("Características")
    plt.xticks(rotation=45)
    plt.legend(title="Cluster")
    plt.show()
