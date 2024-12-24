import sys
import os
#----- Add the project root to the PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import yaml
import pickle
import numpy as np
from src.scripts.dataloader import DataLoader
from src.scripts.preprocessor import Preprocessor
from src.model.kprototypes import KPrototypesCustom
from src.scripts.evaluation import find_best_k
from src.scripts.visualization import visualize_clusters, analyze_cluster_characteristics
from utils.config_loader import load_config
from utils.logger import setup_logger


def main():
    #----- Set up the logger
    logger = setup_logger()

    #----- Load configurations
    config = load_config()
    logger.info("Configurations loaded successfully.")

    #----- Load data
    loader = DataLoader(config['data']['url'])
    data = loader.load_data()
    logger.info(f"Data loaded: {data.shape[0]} rows, {data.shape[1]} columns.")

    #----- Preprocess data
    preprocessor = Preprocessor(
        categorical_cols=config['data']['categorical_cols'],
        numeric_cols=config['data']['numeric_cols']
    )
    X_num_scaled, X_cat_encoded = preprocessor.fit_transform(data)
    logger.info("Preprocessing completed.")

    #----- Determine the best value of k if configured
    if config['clustering']['use_best_k']:
        logger.info("Searching for the best k value...")
        best_k_results = find_best_k(
            X_num_scaled,
            X_cat_encoded,
            max_clusters=config['evaluation']['max_clusters'],
            gamma=config['evaluation']['gamma'],
            penalty_alpha=config['evaluation']['penalty_alpha']
        )
        n_clusters = best_k_results['k_values'][np.argmax(best_k_results['combined_metric'])]
        logger.info(f"Best k value found: {n_clusters}")
    else:
        n_clusters = config['clustering']['n_clusters']
        logger.info(f"Using the predefined number of clusters in configuration: {n_clusters}")

    #----- Train the model
    model = KPrototypesCustom(n_clusters=n_clusters, gamma=config['evaluation']['gamma'])
    model.fit(X_num_scaled.values, X_cat_encoded.values)
    logger.info("Model training completed.")

    #----- Save the model
    with open("data/output/kprototypes_model.pkl", "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Trained model saved at 'data/output/kprototypes_model.pkl'.")

    #----- Generate predictions
    labels = model.predict(X_num_scaled.values, X_cat_encoded.values)
    data['Cluster'] = labels
    logger.info("Predictions generated and clusters assigned.")

    #----- Visualize the clusters
    logger.info("Generating visualization plots...")
    visualize_clusters(data, X_num_scaled, X_cat_encoded, labels)

    #----- Analyze features by cluster
    analyze_cluster_characteristics(data, config['data']['numeric_cols'])
    logger.info("Cluster feature analysis completed.")

    logger.info("Pipeline executed successfully.")

if __name__ == "__main__":
    main()
