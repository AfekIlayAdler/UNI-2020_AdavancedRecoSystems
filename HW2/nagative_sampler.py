import numpy as np
import pandas as pd
from tqdm import tqdm

from HW2.config import ITEM_COL, USER_COL, INTERNAL_DATA_DIR, NEGATIVE_SAMPLES_FILE_NAME, N_NEGATIVE_SAMPLES_SETS, LOAD_NEGATIVE


class NegativeSampler:
    """ choose the negative sample proportionally to their item strength
    can get inverse parameter such that it will become InversePopularityNegativeSampler"""

    def __init__(self, item_probabilities, method='popularity'):
        self.method = method
        self.item_probabilities = item_probabilities
        self.path = INTERNAL_DATA_DIR / f"{NEGATIVE_SAMPLES_FILE_NAME}.csv"
        self.positive_samples = None

    def adjust_probabilities(self, p):
        p = p / p.sum()  # renormalize for popularity
        if self.method == 'uniform':
            p = pd.Series(1 / p.size, p.index)
        elif self.method == 'inverse_popularity':
            p = (1 / p)
            p = p / p.sum()
        return p

    def _create_and_save_negative_samples(self, data):  # data is train data
        assert self.method in ['popularity', 'inverse_popularity', 'uniform'], 'negative sampling method not supported'
        print("Creating negative samples")
        data.sort_values(by=USER_COL, inplace=True)
        self.positive_samples = data
        unique_items = set(data[ITEM_COL].unique())
        for user in tqdm(data[USER_COL].unique(), total=len(data[USER_COL].unique())):
            user_unique_items = data[data[USER_COL] == user][ITEM_COL].unique()
            user_items_did_not_rank = list(unique_items.difference(user_unique_items))
            p = pd.Series({i: self.item_probabilities[i] for i in user_items_did_not_rank})
            p = self.adjust_probabilities(p)
            user_negative_samples = pd.DataFrame()
            user_n_items = len(user_unique_items)
            replace_or_not = len(user_items_did_not_rank) < user_n_items
            for i in range(1, N_NEGATIVE_SAMPLES_SETS):
                user_negative_samples[i] = np.random.choice(p.index,
                                                            size=user_n_items,
                                                            replace=replace_or_not, p=p.values)
            user_negative_samples.to_csv(self.path, mode='a', header=False, index=False)

    def create_negative_samples(self, data):
        if self.path.exists() and LOAD_NEGATIVE:
            self.positive_samples = data
        else:
            empty_df = pd.DataFrame(columns=[i for i in range(1, N_NEGATIVE_SAMPLES_SETS)])  # create file
            empty_df.to_csv(self.path, index=False)
            self._create_and_save_negative_samples(data)

    def get(self, epoch):
        negative_samples = pd.read_csv(self.path, usecols=[epoch - 1])
        train = self.positive_samples.copy()
        train['negative_samples'] = negative_samples
        return train.values
