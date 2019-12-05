import pandas as pd
from scipy.sparse import csr_matrix
from HW1.config import USERS_COL_NAME, ITEMS_COL_NAME, USER_COL, ITEM_COL, RATING_COL_NAME


class Config:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def add_attributes(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def preprocess_for_mf(train, validation):
    train[USER_COL] = pd.factorize(train[USERS_COL_NAME])[0]
    train[ITEM_COL] = pd.factorize(train[ITEMS_COL_NAME])[0]
    user_map = train[[USER_COL, USERS_COL_NAME]].drop_duplicates()
    user_map = user_map.set_index(USERS_COL_NAME).to_dict()[USER_COL]
    item_map = train[[ITEM_COL, ITEMS_COL_NAME]].drop_duplicates()
    item_map = item_map.set_index(ITEMS_COL_NAME).to_dict()[ITEM_COL]
    cols_to_use = [USER_COL, ITEM_COL, RATING_COL_NAME]
    validation = validation[
        validation[USERS_COL_NAME].isin(train[USERS_COL_NAME].unique()) & validation[ITEMS_COL_NAME].isin(
            train[ITEMS_COL_NAME].unique())]
    validation[USER_COL] = validation[USERS_COL_NAME].map(user_map)
    validation[ITEM_COL] = validation[ITEMS_COL_NAME].map(item_map)
    return train[cols_to_use], validation[cols_to_use], user_map, item_map


def get_dict(group, col_name):
    if col_name==ITEMS_COL_NAME:
        return dict(zip(group[ITEMS_COL_NAME], group[RATING_COL_NAME]))
    else:
        return dict(zip(group[USERS_COL_NAME], group[RATING_COL_NAME]))

def to_sparse_matrix(df, row, col):
    mat_csr = csr_matrix((df[RATING_COL_NAME], (df[row], df[col])))
    return mat_csr

def preprocess_for_mf_sgd_als(train, validation):
    train, validation, user_map, item_map = preprocess_for_mf(train, validation)
    user_item_sparse_matrix = to_sparse_matrix(train, USER_COL, ITEM_COL)
    user_item_sparse_matrix.data = user_item_sparse_matrix.data.astype(float)
    return user_item_sparse_matrix, train, validation,  user_map, item_map