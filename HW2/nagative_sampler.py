import numpy as np
import pandas as pd

from HW2.config import RANK_COL, ITEM_COL, USER_COL, MF_WEIGHT_DIR, MF_LOAD_TRAIN_VALIDATION, \
    NEGATIVE_SAMPLES_FILE_NAME, MF_LOAD_NEGATIVE_SAMPLES, MF_SAVE_NEGATIVE_SAMPLES, BPR, POSITIVE_COL, negative_col
from tqdm import tqdm


class NegativeSampler:
    """ choose the negative sample proportionally to their item strength
    can get inverse parameter such that it will become InversePopularityNegativeSampler"""

    def __init__(self, item_probabilities, sample_proportion=1, method='popularity'):
        self.sample_proportion = sample_proportion
        self.method = method
        self.item_probabilities = item_probabilities

    def adjust_probabilities(self, p):
        p = p / p.sum()  # renormalize for popularity
        if self.method == 'uniform':
            p = pd.Series(1 / p.size, p.index)
        elif self.method == 'inverse_popularity':
            p = (1 / p)
            p = p / p.sum()
        return p

    def add_negative_samples(self, data):
        assert self.method in ['popularity', 'inverse_popularity', 'uniform'], 'negative sampling method not supported'
        print("Creating negative samples")
        data.sort_values(by=USER_COL, inplace=True)
        unique_items = set(data[ITEM_COL].unique())
        unique_users = data[USER_COL].unique()
        data[RANK_COL] = 1
        col_order = [USER_COL, ITEM_COL, RANK_COL]
        df_list = []
        for user in tqdm(unique_users, total=len(unique_users)):
            user_unique_items = data[data[USER_COL] == user][ITEM_COL].unique()
            user_items_did_not_rank = list(unique_items.difference(user_unique_items))
            p = pd.Series({i: self.item_probabilities[i] for i in user_items_did_not_rank})
            p = self.adjust_probabilities(p)
            df = pd.DataFrame()
            # boolean indicator - sample with replacement or not by #items user rank*2>total_items
            replace_or_not = len(user_items_did_not_rank) < len(user_unique_items) * self.sample_proportion
            df[ITEM_COL] = np.random.choice(p.index,
                                            size=int(len(user_unique_items) * self.sample_proportion),
                                            replace=replace_or_not, p=p.values)
            df[USER_COL] = user
            df[RANK_COL] = 0
            df_list.append(df)
        if BPR:
            df = pd.concat(df_list)
            df.sort_values(by=USER_COL, inplace=True)
            data = data.rename(columns={ITEM_COL: POSITIVE_COL})
            data[negative_col] = df[ITEM_COL].values
            data = data[[USER_COL, POSITIVE_COL, negative_col]]
        else:
            df = pd.concat(df_list)
            data = pd.concat([data, df[col_order]])
        return data

    def get(self, train, epoch):
        file_name = f"{NEGATIVE_SAMPLES_FILE_NAME}_{self.method}_epoch_{epoch}.csv"
        path = MF_WEIGHT_DIR / file_name
        if path.exists() and MF_LOAD_NEGATIVE_SAMPLES:
            negative_samples = pd.read_csv(path)
        else:
            negative_samples = self.add_negative_samples(train)
        if MF_SAVE_NEGATIVE_SAMPLES:
            negative_samples.to_csv(path, index=False)
        return negative_samples.values
