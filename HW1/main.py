import pandas as pd

from HW1.MatrixFactorizationModelSGD import MatrixFactorizationWithBiasesSGD
from HW1.MartixFactorizationModelALS import MatrixFactorizationWithBiasesALS
from HW1.config import TRAIN_PATH, VALIDATION_PATH
from HW1.utils import preprocess_for_mf_sgd_als, Config


def main():
    train, validation = pd.read_csv(TRAIN_PATH), pd.read_csv(VALIDATION_PATH)
    user_item_sparse_matrix, train, validation, user_map, item_map = preprocess_for_mf_sgd_als(train, validation)
    n_users = len(user_map)
    n_items = len(item_map)
    results = []
    for i in [16]:  # range(4, 32, 2):
        config = Config(
            ratings=user_item_sparse_matrix,
            hidden_dimension=i, lr=0.01,
            l2_users=0.09,
            l2_items=0.09,
            l2_users_bias=0.09,
            l2_items_bias=0.09,
            epochs=20,
            bias_epochs=1,
            n_users=n_users, n_items=n_items, seed=1)

        # mf = MatrixFactorizationWithBiasesSGD(config)
        # mf.fit(train.values, validation.values, user_map, item_map)

        mf = MatrixFactorizationWithBiasesALS(config)
        mf.fit(train.values, validation.values, user_map, item_map)

        results.append(mf.get_results())
    pd.concat(results, axis=1).to_csv('elbow.csv')


if __name__ == "__main__":
    main()
