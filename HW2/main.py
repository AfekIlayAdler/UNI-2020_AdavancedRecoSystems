import pandas as pd

from HW2.config import SEED, ONE_CLASS_MF_WEIGHT_DIR
from HW2.config import TRAIN_PATH
from HW2.optimization_objects import Config
from HW2.utils import preprocess_for_mf, create_directories
from MatrixFactorizationModelSGD import OneClassMatrixFactorizationWithBiasesSGD
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
    epochs=25,
    bias_epochs=5,
    seed=SEED)

if __name__ == "__main__":
    train = pd.read_csv(TRAIN_PATH, nrows=5000)
    create_directories([ONE_CLASS_MF_WEIGHT_DIR])
    train, user_map, item_map = preprocess_for_mf(train)
    mf = OneClassMatrixFactorizationWithBiasesSGD(CONFIG)
    negative_sampler = NegativeSampler()
    validation_creator = ValidationCreator()
    train, validation = validation_creator.get(train)
    mf.fit(train, user_map, item_map, negative_sampler, validation)
