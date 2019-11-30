import pandas as pd
from scipy import sparse
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

def get_train_val_dicts(data):
    dict_users = {}
    groupby_user = data.groupby([USERS_COL_NAME])
    for group_name, group in groupby_user:
        dict_users[group_name] = get_dict(group, ITEMS_COL_NAME)
    dict_items = {}
    groupby_item = data.groupby([ITEMS_COL_NAME])
    for group_name, group in groupby_item:
        dict_items[group_name] = get_dict(group, USERS_COL_NAME)
    return dict_users, dict_items


def to_sparse_matrix(df, row, col):
    mat_csr = csr_matrix((df[RATING_COL_NAME], (df[row], df[col])))
    return mat_csr

def preprocess_for_mf_als(train, validation):
    #item_matrix = train.pivot_table(index=USERS_COL_NAME, columns=ITEMS_COL_NAME, values=RATING_COL_NAME)
    #item_matrix.head()
    # train_dict_users, train_dict_items = get_train_val_dicts(train)
    # val_dict_users, val_dict_items = get_train_val_dicts(validation)
    # return train_dict_users, train_dict_items,val_dict_users, val_dict_items, user_map, item_map

    train, validation, user_map, item_map = preprocess_for_mf(train, validation)
    user_item_sparse_matrix = to_sparse_matrix(train, USER_COL, ITEM_COL)
    item_user_sparse_matrix = to_sparse_matrix(train, ITEM_COL, USER_COL)
    return user_item_sparse_matrix, item_user_sparse_matrix, user_map, item_map