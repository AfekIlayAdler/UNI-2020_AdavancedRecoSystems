SEED = 5
from pathlib import Path

#one_class_mf save configurations
VALIDATION_FILE_NAME = 'one_class_mf_validation.csv'
TRAIN_FILE_NAME = 'one_class_mf_train.csv'
ONE_CLASS_MF_WEIGHT_DIR = Path('one_class_mf_weights_sgd')
NEGATIVE_SAMPLES_FILE_NAME = 'negative_samples'
ONE_CLASS_MF_SAVE_TRAIN_VALIDATION = True
ONE_CLASS_MF_LOAD_TRAIN_VALIDATION = True
ONE_CLASS_MF_SAVE_NEGATIVE_SAMPLES = True
ONE_CLASS_MF_LOAD_NEGATIVE_SAMPLES = True



TRAIN_PATH = 'data/Train.csv'
RANDOM_TEST_PATH = 'data/RandomTest.csv'
RANDOM_TEST_COL_NAME1 = 'Item1'
RANDOM_TEST_COL_NAME2 = 'Item2'
USERS_COL_NAME = 'UserID'
ITEMS_COL_NAME = 'ItemID'
# internal
USER_COL = 'user'
ITEM_COL = 'item'
RANK_COL = 'rank'
# TEST_PATH = 'data/Test.csv'

# RATING_COL_NAME = 'Ratings_Rating'

# model_name = 'sgd' if SGD else 'als'
# CHECKPOINT_NAME = f"./checkpoint_{model_name}.pkl"
# HYPER_PARAM_FILE_NAME = f"HyperParamResult_{model_name}.pkl"
