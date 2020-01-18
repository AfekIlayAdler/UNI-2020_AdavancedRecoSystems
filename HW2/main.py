import pandas as pd

from HW2.config import INTERNAL_DATA_DIR, RESULT_DIR, RESULT_FILE_NAME, CONFIG
from HW2.config import TRAIN_PATH
from HW2.utils import preprocess_for_mf, create_directories
from MatrixFactorizationModelSGD_BPR import BPRMatrixFactorizationWithBiasesSGD
from validation_creator import ValidationCreator

if __name__ == "__main__":
    train = pd.read_csv(TRAIN_PATH)
    create_directories([INTERNAL_DATA_DIR, RESULT_DIR])
    train, user_map, item_map = preprocess_for_mf(train)
    config = CONFIG
    validation_creator = ValidationCreator(config.validation_creator_sampler_type)
    train, validation = validation_creator.get(train)
    config.add_attributes(n_users=len(user_map), n_items=len(item_map))
    mf = BPRMatrixFactorizationWithBiasesSGD(config)
    mf.fit(train, user_map, item_map, validation)
    mf.get_results().to_csv(RESULT_DIR / RESULT_FILE_NAME, index=False)
