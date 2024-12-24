import pytest
import pandas as pd
from src.scripts.dataloader import DataLoader

#----- Fixture to set up test data
@pytest.fixture
def valid_url():
    return "https://archive.ics.uci.edu/ml/machine-learning-databases/00292/Wholesale%20customers%20data.csv"

@pytest.fixture
def invalid_path():
    return "invalid/path/to/file.csv"

#----- Test to load data from a valid URL
def test_load_data_from_url(valid_url):
    loader = DataLoader(valid_url)
    data = loader.load_data()
    assert isinstance(data, pd.DataFrame), "Data is not a DataFrame"
    assert len(data) > 0, "The DataFrame should have rows"

#----- Test to handle an invalid path
def test_load_data_invalid_path(invalid_path):
    loader = DataLoader(invalid_path)
    with pytest.raises(FileNotFoundError):
        loader.load_data()
