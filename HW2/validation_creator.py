from config import RANDOM_TEST_PATH, RANDOM_TEST_COL_NAME1, USERS_COL_NAME, USER_COL, ITEM_COL, RANK_COL, \
    MF_WEIGHT_DIR, VALIDATION_FILE_NAME, MF_LOAD_TRAIN_VALIDATION, TRAIN_FILE_NAME, \
    MF_SAVE_TRAIN_VALIDATION, BPR, positive_col, negative_col
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle


class ValidationCreator:
    def __init__(self, method='popularity'):
        self.method = method
        self.train_path = MF_WEIGHT_DIR / TRAIN_FILE_NAME
        self.validation_path = MF_WEIGHT_DIR / VALIDATION_FILE_NAME

    def adjust_probabilities(self, p):
        p = p / p.sum()  # renormalize - so sum of probabilities will sum to 1
        if self.method == 'uniform':
            p = pd.Series(1 / p.size, p.index)
        return p

    def choose_items(self, did_rank, did_not_rank, item_probabilities):
        did_rank, did_not_rank = list(did_rank), list(did_not_rank)
        positive_and_negative = []
        for i, items in enumerate([did_rank, did_not_rank]):
            p = pd.Series({i: item_probabilities[i] for i in items})
            # p = p / p.sum()
            p = self.adjust_probabilities(p)
            positive_and_negative.append(np.random.choice(p.index, p=p.values, size=1)[0])
        return positive_and_negative

    def split(self, data):
        assert self.method in ['popularity', 'uniform'], 'negative sampling method not supported'
        item_probabilities = (data[ITEM_COL].value_counts() / data.shape[0]).to_dict()
        unique_items = set(data[ITEM_COL].unique())
        validation = []
        print("Creating train & validation sets")
        train = data.copy()
        for user in tqdm(data[USER_COL].unique(), total=len(data[USER_COL].unique())):
            user_unique_items = set(data[data[USER_COL] == user][ITEM_COL].unique())
            user_items_did_not_rank = unique_items.difference(user_unique_items)
            # choose from train sample of item the rank and didnt rank, add to validation and remove from train
            one_did_rank, one_did_not_rank = self.choose_items(user_unique_items, user_items_did_not_rank,
                                                               item_probabilities)
            validation.append([user, one_did_rank, one_did_not_rank])
            train = train[(train[USER_COL] != user) | ((train[USER_COL] == user) & (train[ITEM_COL] != one_did_rank))]
        validation = pd.DataFrame(validation, columns=[USER_COL, positive_col, negative_col])
        return train, validation

    def get(self, train):
        if self.train_path.exists() and MF_LOAD_TRAIN_VALIDATION:
            train, validation = pd.read_csv(self.train_path), pd.read_csv(self.validation_path)
        else:
            train, validation = self.split(train)
        if MF_SAVE_TRAIN_VALIDATION:
            train.to_csv(self.train_path, index=False)
            validation.to_csv(self.validation_path, index=False)
        return train, validation


def create_validation_two_columns(data):
    columns = [USER_COL, ITEM_COL, RANK_COL]
    df1 = data[[USER_COL, positive_col]]
    df1[RANK_COL] = 1
    df1.columns = columns
    df2 = data[[USER_COL, negative_col]]
    df2[RANK_COL] = 0
    df2.columns = columns
    return pd.concat([df1, df2]).values, data
