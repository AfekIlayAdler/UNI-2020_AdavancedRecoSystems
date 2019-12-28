import numpy as np
import pandas as pd

from HW2.config import RANK_COL, ITEM_COl


class NegativeSampler():
    def __init__(self, sample_propotion):
        self.sample_propotion = sample_propotion

    def add_negative_samples(self, data):
        raise NotImplementedError


class UniformNegativeSampler(NegativeSampler):
    def __init__(self, sample_propotion = 1):
        super().__init__(sample_propotion)

    def add_negative_samples(self, data):
        unique_items = set(data[ITEM_COl].unique())
        unique_users = data[USER_COl].unique()
        data[RANK_COL] = 1
        for user in unique_users:
            user_unique_items = data[data[USER_COl] == user][ITEM_COl].unique()
            user_items_didnot_rank = unique_items.difference(user_unique_items)
            df = pd.DataFrame()
            df[ITEM_COl] = np.random.choice(list(user_items_didnot_rank),
                                                  size=int(len(user_unique_items) * self.sample_propotion),
                                                  replace=False)
            df[USER_COl] = user
            df[RANK_COL] = 0
            data = pd.concat([data, df])
        return data.values
