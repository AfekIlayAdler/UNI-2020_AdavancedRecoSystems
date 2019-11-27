import pandas as pd

from MatrixFactorizationModelSGD import MatrixFactorizationWithBiases
from config import TRAIN_PATH, VALIDATION_PATH
from utils import preprocess_for_mf, Config


def main():
    train, validation = pd.read_csv(TRAIN_PATH), pd.read_csv(VALIDATION_PATH)
    train, validation, user_map, item_map = preprocess_for_mf(train, validation)
    n_users = len(user_map)
    n_items = len(item_map)

    config = Config(
        hidden_dimension=15, lr=10 ** (-2),
        l2_users=0.002,
        l2_items=0.002,
        l2_users_bias=0.002,
        l2_items_bias=0.002,
        epochs=12,
        bias_epochs=4,
        n_users=n_users, n_items=n_items)

    mf = MatrixFactorizationWithBiases(config)
    mf.fit(train.values, validation.values, user_map, item_map)


if __name__ == "__main__":
    main()
