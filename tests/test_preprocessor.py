import pytest
import pandas as pd
from src.scripts.preprocessor import Preprocessor

#----- Fixture to set up the test data
@pytest.fixture
def test_data():
    data = pd.DataFrame({
        'Categorical': ['A', 'B', 'A', 'C'],
        'Numeric': [1.0, 2.5, 3.2, 4.8]
    })
    categorical_cols = ['Categorical']
    numeric_cols = ['Numeric']
    return data, categorical_cols, numeric_cols

#----- Test to verify preprocessing
def test_fit_transform(test_data):
    data, categorical_cols, numeric_cols = test_data
    preprocessor = Preprocessor(categorical_cols, numeric_cols)

    X_num_scaled, X_cat_encoded = preprocessor.fit_transform(data)

    #----- Verify that the scaled data has the correct dimensions
    assert X_num_scaled.shape[1] == len(numeric_cols), "Incorrect number of numeric columns"
    assert X_cat_encoded.shape[1] == len(categorical_cols), "Incorrect number of categorical columns"

    #----- End of test_fit_transform
