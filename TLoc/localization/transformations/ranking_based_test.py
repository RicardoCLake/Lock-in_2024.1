import unittest

import pandas as pd
import numpy as np

from localization.transformations.ranking_based import RankingBased


def test_ranking_based_representation():
    mock_data = np.array([
            [0, 0, 5, 0, 4, 0],
            [97, 8, 50, 0, 100, 20],
            [97, 8, 50, 5, 100, 20],
        ], dtype=float)
    result = RankingBased().transform(mock_data)
    expected_result = np.array([
        [-1, -1, 1, -1, 2, -1],
        [2, 5, 3, -1, 1, 4],
        [2, 5, 3, 6, 1, 4],
    ])
    assert np.array_equal(result, expected_result)
