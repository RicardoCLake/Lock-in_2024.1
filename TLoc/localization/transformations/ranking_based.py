import numpy as np

class RankingBased:
    def __init__(self):
        None

    def transform(self, X, null_element=0, default_value=-1):
        return np.apply_along_axis(lambda row: self.get_ranking(row, null_element, default_value), 1, X)

    def get_ranking(self, rss, null_element, default_value=-1):
        # the strongest to the weakest AP
        ix = np.argsort(rss)[::-1]
        # index of first null element
        first_null = rss.shape[0]
        null_ix = np.argwhere(rss[ix] == null_element)
        if len(null_ix) > 0:
            first_null = int(null_ix[0])
        ranking = np.zeros(len(ix)) + default_value
        for i in range(first_null):
            ranking[ix[i]] = i + 1
        return ranking
