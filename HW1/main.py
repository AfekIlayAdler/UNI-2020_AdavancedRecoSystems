from MatrixFactorizationModel import MatrixFactorizationWithBiases
from config import TRAIN_URL, USERS_COL_NAME, ITEMS_COL_NAME, USER_COL, ITEM_COL, RATING_COL_NAME
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


class Config:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def add_attributes(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def main():
    df = pd.read_csv(TRAIN_URL)
    df[USER_COL] = pd.factorize(df[USERS_COL_NAME])[0]
    df[ITEM_COL] = pd.factorize(df[ITEMS_COL_NAME])[0]
    n_users = len(df[USER_COL].unique())
    n_items = len(df[ITEM_COL].unique())
    df = shuffle(df)
    df = df[[USER_COL, ITEM_COL, RATING_COL_NAME]]
    train, validation = train_test_split(df, test_size=0.2, random_state=42)

    config = Config(
        hidden_dimension=15, lr=10 ** (-2),
        l2_users=0.002,
        l2_items=0.002,
        l2_users_bias=0.002,
        l2_items_bias=0.002,
        epochs=12,
        bias_epochs=4)

    config.add_attributes(n_users=n_users, n_items=n_items)
    mf = MatrixFactorizationWithBiases(config)
    mf.fit(train.values, validation.values)


if __name__ == "__main__":
    main()
