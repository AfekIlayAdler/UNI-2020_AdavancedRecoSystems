import numpy as np
import pandas as pd

from HW2.config import RANK_COL, ITEM_COL, USER_COL, ONE_CLASS_MF_WEIGHT_DIR, ONE_CLASS_MF_LOAD_TRAIN_VALIDATION, \
    NEGATIVE_SAMPLES_FILE_NAME, ONE_CLASS_MF_LOAD_NEGATIVE_SAMPLES, ONE_CLASS_MF_SAVE_NEGATIVE_SAMPLES
from tqdm import tqdm


class NegativeSampler:
    """ choose the negative sample proportionally to their item strength
    can get inverse parameter such that it will become InversePopularityNegativeSampler"""

    def __init__(self, sample_proportion=1, method='popularity'):
        self.sample_proportion = sample_proportion
        self.method = method

    def adjust_probabilities(self, p):
        p = p / p.sum()  # renormalize
        if self.method == 'uniform':
            p = pd.Series(1 / p.size, p.index)
        elif self.method == 'inverse_popularity':
            p = (1 / p)
            p = p / p.sum()
        return p

    def add_negative_samples(self, data):
        assert self.method in ['popularity', 'inverse_popularity', 'uniform'], 'negative sampling method not supported'
        print("Creating negative samples")
        item_probabilities = (data[ITEM_COL].value_counts() / data.shape[0]).to_dict()
        unique_items = set(data[ITEM_COL].unique())
        unique_users = data[USER_COL].unique()
        data[RANK_COL] = 1
        col_order = [USER_COL, ITEM_COL, RANK_COL]
        for user in tqdm(unique_users, total=len(unique_users)):
            user_unique_items = data[data[USER_COL] == user][ITEM_COL].unique()
            user_items_did_not_rank = list(unique_items.difference(user_unique_items))
            p = pd.Series({i: item_probabilities[i] for i in user_items_did_not_rank})
            p = self.adjust_probabilities(p)
            df = pd.DataFrame()
            replace_or_not = len(user_items_did_not_rank) < len(user_unique_items) * self.sample_proportion
            df[ITEM_COL] = np.random.choice(p.index,
                                            size=int(len(user_unique_items) * self.sample_proportion),
                                            replace=replace_or_not, p=p.values)
            df[USER_COL] = user
            df[RANK_COL] = 0
            data = pd.concat([data, df[col_order]])
        return data

    def get(self, train, epoch):
        file_name = f"{NEGATIVE_SAMPLES_FILE_NAME}_{self.method}_epoch_{epoch}.csv"
        path = ONE_CLASS_MF_WEIGHT_DIR / file_name
        if path.exists() and ONE_CLASS_MF_LOAD_NEGATIVE_SAMPLES:
            negative_samples = pd.read_csv(path)
        else:
            negative_samples = self.add_negative_samples(train)
        if ONE_CLASS_MF_SAVE_NEGATIVE_SAMPLES:
            negative_samples.to_csv(path, index=False)
        return negative_samples.values
