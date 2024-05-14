import pandas as pd
import numpy as np

from localization.augmentation.augmentation import PowerAverages
from localization.dataset import Dataset


class MockDataset(Dataset):
    def __init__(self, df_training, df_validation):
        self.df_validation = df_validation
        self.df_training = df_training
        self.location_columns = ['LONGITUDE', 'LATITUDE', 'FLOOR', 'BUILDINGID']
        self.input_columns = ['WAP001', 'WAP002', 'WAP003']


def test_get_power_average():
    mock_data = pd.DataFrame(
        np.array([
            # pos 0, only WAP002 appears
            [0, 1, 1, 0, 0, 1, 0],
            [0, 3, 0, 0, 0, 1, 0],
            [0, 2, 0, 0, 0, 1, 0],
            # pos 1, APS 001 and 003 appear
            [1, 0, 2, 10, 0, 0, 1],
            [2, 2, 10, 10, 0, 0, 1]
        ], dtype=float),
        columns=['WAP001', 'WAP002', 'WAP003', 'LONGITUDE', 'LATITUDE', 'FLOOR', 'BUILDINGID'])
    expected_return = pd.DataFrame(
        np.array([
            # pos 0, only WAP002 appears
            [np.nan, 2, np.nan],
            # pos 1, APS 001 and 003 appear
            [1.5, np.nan, 6],
        ]),
        columns=['WAP001', 'WAP002', 'WAP003'])

    expected_positions = pd.DataFrame(
        np.array([
            [0, 0, 1, 0],
            [10, 0, 0, 1],
        ], dtype=float),
        columns=['LONGITUDE', 'LATITUDE', 'FLOOR', 'BUILDINGID'])

    dataset = MockDataset(mock_data, None)
    power_averages = PowerAverages(dataset)

    assert power_averages.power_averages_per_position.equals(expected_return)
    assert power_averages.distinct_positions.equals(expected_positions)
