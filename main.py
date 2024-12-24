import yaml
from dataloader import DataLoader
from preprocessor import Preprocessor
from model.kprototypes import KPrototypesCustom
from evaluation import find_best_k
from visualization import visualize_clusters, analyze_cluster_characteristics
import numpy as np

def load_config(config_path="config.yaml"):
    """
    Carga las configuraciones desde un archivo YAML.
    :param config_path: Ruta al archivo de configuración.
    :return: Diccionario con las configuraciones cargadas.
    """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def main():
    # Cargar configuraciones
    config = load_config("config/config.yaml")

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00292/Wholesale%20customers%20data.csv"

    # Cargar datos
    loader = DataLoader(url)
    data = loader.load_data()

    # Definir columnas categóricas y numéricas
    categorical_cols = ['Channel', 'Region']
    numeric_cols = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']

    # Preprocesamiento
    preprocessor = Preprocessor(categorical_cols, numeric_cols)
    X_num_scaled, X_cat_encoded = preprocessor.fit_transform(data)

    # Determinar número de clusters
    if config['clustering']['use_best_k']:
        print("Buscando el mejor valor de k...")
        
        best_k_results = find_best_k(
        X_num_scaled,
        X_cat_encoded,
        max_clusters=config['evaluation']['max_clusters'],
        gamma=config['evaluation']['gamma'],
        penalty_alpha=config['evaluation']['penalty_alpha'])  # Incluido desde el archivo de configuración
        
        # Usar la métrica combinada para determinar el mejor k
        n_clusters = best_k_results['k_values'][np.argmax(best_k_results['combined_metric'])]
        print(f"Mejor valor de k encontrado: {n_clusters}")
    else:
        n_clusters = config['clustering']['n_clusters']
        print(f"Usando número de clusters definido en configuración: {n_clusters}")

    # Entrenamiento del modelo
    kproto = KPrototypesCustom(n_clusters=n_clusters, gamma=config['evaluation']['gamma'])
    kproto.fit(X_num_scaled.values, X_cat_encoded.values)

    # Predicción
    labels = kproto.predict(X_num_scaled.values, X_cat_encoded.values)

    # Agregar clusters al DataFrame
    data['Cluster'] = labels
    print(data.head())

    # Visualizar los clusters en un espacio reducido
    visualize_clusters(data, X_num_scaled, X_cat_encoded, labels)

    # Analizar las características numéricas por cluster
    analyze_cluster_characteristics(data, numeric_cols)

if __name__ == "__main__":
    main()
