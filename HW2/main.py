import pandas as pd

from HW2.config import SEED, ONE_CLASS_MF_WEIGHT_DIR, BPR, RESULT_DIR, ONE_CLASS_MF_RESULT_FILE_NAME
from HW2.config import TRAIN_PATH
from HW2.optimization_objects import Config
from HW2.utils import preprocess_for_mf, create_directories
from MatrixFactorizationModelSGD import OneClassMatrixFactorizationWithBiasesSGD
from nagative_sampler import NegativeSampler
from validation_creator import ValidationCreator

CONFIG = Config(
    lr=0.001,
    print_metrics=True,
    beta=0.9,
    hidden_dimension=18,
    l2_users=0.01,
    l2_items=0.01,
    l2_users_bias=0.001,
    l2_items_bias=0.001,
    epochs=10,
    bias_epochs=10,
    seed=SEED)

if __name__ == "__main__":
    train = pd.read_csv(TRAIN_PATH, nrows=10 ** 5)
    create_directories([ONE_CLASS_MF_WEIGHT_DIR, RESULT_DIR])
    train, user_map, item_map = preprocess_for_mf(train)
    validation_creator = ValidationCreator()
    train, validation = validation_creator.get(train)
    config = CONFIG
    config.add_attributes(n_users=len(user_map), n_items=len(item_map))
    if BPR:
        pass
    else:
        mf = OneClassMatrixFactorizationWithBiasesSGD(config, NegativeSampler())
        mf.fit(train, user_map, item_map, validation)
        mf.get_results().to_csv(RESULT_DIR / ONE_CLASS_MF_RESULT_FILE_NAME, index=False)
