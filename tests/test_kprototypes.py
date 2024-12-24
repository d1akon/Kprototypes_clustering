import pytest
import pandas as pd
from src.model.kprototypes import KPrototypesCustom

#----- Fixture to set up simulated data
@pytest.fixture
def simulated_data():
    X_num = pd.DataFrame({
        'Feature1': [1.0, 2.0, 3.0],
        'Feature2': [4.0, 5.0, 6.0]
    })
    X_cat = pd.DataFrame({
        'Category1': [0, 1, 0],
        'Category2': [1, 0, 1]
    })
    return X_num, X_cat

#----- Test to verify model fitting
def test_fit(simulated_data):
    X_num, X_cat = simulated_data
    model = KPrototypesCustom(n_clusters=2, gamma=1.0)
    model.fit(X_num.values, X_cat.values)

    #----- Verify that the centroids are correct
    assert len(model.centroids[0]) == 2, "Number of numeric centroids does not match"
    assert len(model.centroids[1]) == 2, "Number of categorical centroids does not match"

#----- Test to verify model predictions
def test_predict(simulated_data):
    X_num, X_cat = simulated_data
    model = KPrototypesCustom(n_clusters=2, gamma=1.0)
    model.fit(X_num.values, X_cat.values)
    labels = model.predict(X_num.values, X_cat.values)

    #----- Verify that the number of labels matches the data
    assert len(labels) == len(X_num), "The number of labels does not match the number of rows in the data"
