# 💠 Kprototypes_clustering

### 🔷 Overview

K-Prototypes is a clustering algorithm designed to handle mixed-type data, including both numerical and categorical features. Unlike K-Means, which works only with numerical data, K-Prototypes combines the strengths of K-Means and K-Modes, making it ideal for datasets containing both numerical and categorical variables. 

### 🌀 Key Advantages of K-Prototypes:
- **Handles Mixed Data**: Supports clustering with numerical and categorical features simultaneously.
- **Scalability**: Efficient with large datasets.
- **Flexibility**: The `gamma` parameter balances the influence of numerical and categorical features.

This repository implements K-Prototypes clustering from scratch and demonstrates its application on the Wholesale Customers dataset. It includes utilities for data preprocessing, evaluation, visualization, and logging.


### ☄ Config and Parameters
Update the following parameters in config/config.yaml to customize the dataset and clustering behavior:

#### Dataset Configuration
  - url: The URL or local path to the dataset (e.g., CSV file).
  - categorical_cols: A list of categorical column names in the dataset.
  - numeric_cols: A list of numerical column names in the dataset.
#### Clustering Configuration
  - use_best_k: If true, the algorithm will determine the optimal number of clusters (k) based on metrics. If false, the number of clusters will be set by n_clusters.
  - n_clusters: The fixed number of clusters to use if use_best_k is false.
#### Evaluation Parameters
  - max_clusters: The maximum number of clusters to test when determining the optimal k.
  - gamma: The weight applied to categorical distances in the K-Prototypes algorithm.
  - penalty_alpha: A penalty factor applied to the number of clusters in the combined evaluation metric.

## 🔵 Steps to set up and run the project

### 1. Clone the Repository
Clone the repository to your local machine:
```python
git clone https://github.com/d1akon/Kprototypes_clustering.git
cd Kprototypes_clustering
```

### 2. Create a Virtual Environment
It is recommended to use a virtual environment to manage dependencies:

```python
python -m venv mlops-env
```

  Activate the virtual environment:

* On Windows:

```powershell
.\mlops-env\Scripts\activate
```

* On Linux/MacOS:

```bash
source mlops-env/bin/activate
```

### 3. Install Dependencies
Install the required Python packages:

```python
pip install -r requirements.txt
```

### 4. Update Configuration (Optional)
Modify the ```config/config.yaml``` file to fit your dataset or clustering requirements. 

### 5. Run Tests
Verify the implementation by running the test suite:

```python
pytest tests/
```

### 6. Execute the Training Pipeline
Run the train.py pipeline to preprocess data, train the K-Prototypes model, and generate visualizations:

```python
python src/pipelines/train.py
```

#### The output includes:

* **Logs:** Saved in data/output/train.log.
* **Trained Model:** Saved as data/output/kprototypes_model.pkl.
* **Visualizations:** Shown during execution, including:
* **Cluster visualization (PCA/t-SNE).**
* **Cluster-specific numerical feature analysis.**
