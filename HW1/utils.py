import pandas as pd

from HW1.config import USERS_COL_NAME, ITEMS_COL_NAME, USER_COL, ITEM_COL, RATING_COL_NAME


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


