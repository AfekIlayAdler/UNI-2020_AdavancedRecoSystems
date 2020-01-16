import pandas as pd

from HW2.config import SEED, MF_WEIGHT_DIR, BPR, RESULT_DIR, RESULT_FILE_NAME
from HW2.config import TRAIN_PATH,ITEM_COL
from HW2.optimization_objects import Config
from HW2.utils import preprocess_for_mf, create_directories
from MatrixFactorizationModelSGD import OneClassMatrixFactorizationWithBiasesSGD
from MatrixFactorizationModelSGD_BPR import BPRMatrixFactorizationWithBiasesSGD
from nagative_sampler import NegativeSampler
from validation_creator import ValidationCreator

CONFIG = Config(
    lr=0.01,
    print_metrics=True,
    beta=0.9,
    hidden_dimension=18,
    l2_users=0.01,
    l2_items=0.01,
    l2_users_bias=0.001,
    l2_items_bias=0.001,
    epochs=35,
    bias_epochs=5,
    seed=SEED)

if __name__ == "__main__":
    train = pd.read_csv(TRAIN_PATH, nrows=100000) #, nrows=5000
    create_directories([MF_WEIGHT_DIR, RESULT_DIR])
    train, user_map, item_map = preprocess_for_mf(train)
    validation_creator = ValidationCreator()
    train, validation = validation_creator.get(train)
    # calculate the items popularity
    item_probabilities = (train.iloc[:, 1].value_counts() / train.shape[0]).to_dict()
    config = CONFIG
    config.add_attributes(n_users=len(user_map), n_items=len(item_map))
    if BPR:
        mf = BPRMatrixFactorizationWithBiasesSGD(config, NegativeSampler(item_probabilities, method='popularity'))
    else:
        mf = OneClassMatrixFactorizationWithBiasesSGD(config, NegativeSampler(item_probabilities, method='popularity'))
    mf.fit(train, user_map, item_map, validation)
    mf.get_results().to_csv(RESULT_DIR / RESULT_FILE_NAME, index=False)