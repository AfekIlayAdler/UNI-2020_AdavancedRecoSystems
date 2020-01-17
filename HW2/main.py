import pandas as pd

from HW2.config import SEED, MF_WEIGHT_DIR, RESULT_DIR, RESULT_FILE_NAME
from HW2.config import TRAIN_PATH
from HW2.optimization_objects import Config
from HW2.utils import preprocess_for_mf, create_directories
from MatrixFactorizationModelSGD_BPR import BPRMatrixFactorizationWithBiasesSGD
from validation_creator import ValidationCreator

CONFIG = Config(
    lr=0.025,
    print_metrics=True,
    beta=0.9,
    hidden_dimension=18,
    l2_users=0.01,
    l2_items=0.01,
    l2_items_bias=0.001,
    epochs=35,
    bias_epochs=5,
    seed=SEED,
    negative_sampler_popularity='popularity',
    validation_creator_sampler_popularity='popularity')

if __name__ == "__main__":
    train = pd.read_csv(TRAIN_PATH)
    create_directories([MF_WEIGHT_DIR, RESULT_DIR])
    train, user_map, item_map = preprocess_for_mf(train)
    config = CONFIG
    validation_creator = ValidationCreator(config.validation_creator_sampler_popularity)
    train, validation = validation_creator.get(train)
    # calculate the items popularity
    config.add_attributes(n_users=len(user_map), n_items=len(item_map))
    mf = BPRMatrixFactorizationWithBiasesSGD(config)
    mf.fit(train, user_map, item_map, validation)
    mf.get_results().to_csv(RESULT_DIR / RESULT_FILE_NAME, index=False)
