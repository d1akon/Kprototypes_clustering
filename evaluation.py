import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score, adjusted_rand_score
import matplotlib.pyplot as plt
import seaborn as sns
from model.kprototypes import KPrototypesCustom

def find_elbow_point(inertia):
    """
    Encuentra el punto del codo en la curva de inercia.
    :param inertia: Lista de valores de inercia para cada k.
    :return: Índice del mejor k basado en el método del codo.
    """
    deltas = np.diff(inertia)  # Primera derivada
    second_deltas = np.diff(deltas)  # Segunda derivada
    elbow_point = np.argmax(second_deltas) + 2  # Ajustar índice al rango de k
    return elbow_point


from sklearn.metrics import silhouette_score, davies_bouldin_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, davies_bouldin_score

def find_best_k(X_num_scaled, X_cat_encoded, max_clusters=10, gamma=1.0, penalty_alpha=0.1):
    """
    Encuentra el mejor valor de k utilizando el Silhouette Score, Davies-Bouldin Index y una penalización por clusters elevados.
    :param X_num_scaled: Datos numéricos escalados.
    :param X_cat_encoded: Datos categóricos codificados.
    :param max_clusters: Máximo número de clusters a probar.
    :param gamma: Peso de la distancia categórica.
    :param penalty_alpha: Factor de penalización para valores elevados de k.
    :return: Diccionario con los valores de k, Silhouette Scores, Davies-Bouldin Index y métrica combinada.
    """

    from sklearn.metrics import silhouette_score, davies_bouldin_score

    silhouette_scores = []
    davies_bouldin_scores = []
    inertia = []
    k_values = list(range(2, max_clusters + 1))  # Declarar k_values aquí

    for k in k_values:
        print(f"Ajustando para k = {k}...")

        # Entrenar el modelo con el valor actual de k
        model = KPrototypesCustom(n_clusters=k, gamma=gamma)
        model.fit(X_num_scaled.values, X_cat_encoded.values)

        # Calcular inercia
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

        # Calcular Silhouette Score
        labels = model.predict(X_num_scaled.values, X_cat_encoded.values)
        silhouette = silhouette_score(
            np.hstack((X_num_scaled, X_cat_encoded)), labels, metric='euclidean'
        )
        silhouette_scores.append(silhouette)

        # Calcular Davies-Bouldin Index
        dbi = davies_bouldin_score(np.hstack((X_num_scaled, X_cat_encoded)), labels)
        davies_bouldin_scores.append(dbi)

        print(f"k={k}: Inercia={total_inertia:.2f}, Silhouette Score={silhouette:.4f}, DBI={dbi:.4f}")

    # Métrica combinada con penalización
    combined_metric = [
        silhouette_scores[i] - davies_bouldin_scores[i] - penalty_alpha * k_values[i]
        for i in range(len(silhouette_scores))
    ]
    best_k_combined = k_values[np.argmax(combined_metric)]

    # Graficar los resultados
    plt.figure(figsize=(15, 5))

    # Gráfico de Silhouette Score
    plt.subplot(1, 3, 1)
    plt.plot(k_values, silhouette_scores, marker='o', linestyle='--')
    plt.title('Silhouette Score vs Número de Clusters')
    plt.xlabel('Número de Clusters (k)')
    plt.ylabel('Silhouette Score')

    # Gráfico de Davies-Bouldin Index
    plt.subplot(1, 3, 2)
    plt.plot(k_values, davies_bouldin_scores, marker='o', linestyle='--', color='orange')
    plt.title('Davies-Bouldin Index vs Número de Clusters')
    plt.xlabel('Número de Clusters (k)')
    plt.ylabel('DBI (más bajo es mejor)')

    # Gráfico de Métrica Combinada
    plt.subplot(1, 3, 3)
    plt.plot(k_values, combined_metric, marker='o', linestyle='--', color='green')
    plt.title('Métrica Combinada vs Número de Clusters')
    plt.xlabel('Número de Clusters (k)')
    plt.ylabel('Métrica Combinada')

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
    Calcula el Silhouette Score para evaluar los clusters.
    :param X_num_scaled: Datos numéricos escalados.
    :param X_cat_encoded: Datos categóricos codificados.
    :param labels: Etiquetas de los clusters asignados.
    :return: Silhouette Score.
    """
    X_combined = pd.concat([X_num_scaled, X_cat_encoded], axis=1)
    score = silhouette_score(X_combined, labels, metric='euclidean')
    print(f"Silhouette Score: {score:.4f}")
    return score


def evaluate_inertia(model):
    """
    Retorna la inercia del modelo después del ajuste.
    :param model: Modelo K-Prototypes ajustado.
    :return: Inercia del modelo.
    """
    if model.centroids is None:
        raise ValueError("El modelo no está entrenado.")
    inertia = model.centroids[0].shape[0]  # Ejemplo de inercia
    print(f"Inercia del modelo: {inertia}")
    return inertia


def evaluate_ari(true_labels, predicted_labels):
    """
    Calcula el ARI para evaluar los clusters.
    :param true_labels: Etiquetas verdaderas.
    :param predicted_labels: Etiquetas asignadas por el modelo.
    :return: Adjusted Rand Index.
    """
    score = adjusted_rand_score(true_labels, predicted_labels)
    print(f"Adjusted Rand Index: {score:.4f}")
    return score


def visualize_boxplot(data, numeric_cols):
    """
    Genera un boxplot para analizar cómo varían las características numéricas entre clusters.
    :param data: DataFrame con los datos y la columna de clusters.
    :param numeric_cols: Lista de columnas numéricas.
    """
    for col in numeric_cols:
        plt.figure(figsize=(8, 4))
        sns.boxplot(data=data, x='Cluster', y=col, palette='Set3')
        plt.title(f'Distribución de {col} por Cluster')
        plt.show()


def elbow_method(X_num_scaled, X_cat_encoded, max_clusters=10, gamma=1.0):
    """
    Utiliza el Método del Codo para encontrar el número óptimo de clusters.
    :param X_num_scaled: Datos numéricos escalados.
    :param X_cat_encoded: Datos categóricos codificados.
    :param max_clusters: Número máximo de clusters a probar.
    :param gamma: Peso para la distancia categórica.
    """
    from projects.projects_github.Kprototypes_clustering.model.kprototypes import KPrototypesCustom

    inertia = []
    for k in range(2, max_clusters + 1):
        print(f"Calculando para k = {k}...")
        model = KPrototypesCustom(n_clusters=k, gamma=gamma)
        model.fit(X_num_scaled.values, X_cat_encoded.values)
        inertia.append(model.centroids[0].shape[0])  # Ejemplo de inercia

    # Graficar inercia
    plt.figure(figsize=(8, 4))
    plt.plot(range(2, max_clusters + 1), inertia, marker='o', linestyle='--')
    plt.xlabel('Número de Clusters')
    plt.ylabel('Inercia')
    plt.title('Método del Codo')
    plt.show()
