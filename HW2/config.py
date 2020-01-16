from pathlib import Path

BPR = True
SEED = 5

# results
ONE_CLASS_MF_RESULT_FILE_NAME = 'one_class_mf_run_results.csv'
BPR_MF_RESULT_FILE_NAME = 'bpr_mf_run_results.csv'
NEGATIVE_SAMPLES_FILE_NAME = 'negative_samples'

MF_SAVE_TRAIN_VALIDATION = True
MF_LOAD_TRAIN_VALIDATION = False
MF_SAVE_NEGATIVE_SAMPLES = True
MF_LOAD_NEGATIVE_SAMPLES = False

if BPR:
    RESULT_FILE_NAME = BPR_MF_RESULT_FILE_NAME
    MF_WEIGHT_DIR = Path('bpr_mf_weights_sgd')
    VALIDATION_FILE_NAME = 'bpr_mf_validation.csv'
    TRAIN_FILE_NAME = 'bpr_class_mf_train.csv'
    RESULT_DIR = Path(r'results/one_class')
else:
    # one_class_mf save configurations
    RESULT_FILE_NAME = ONE_CLASS_MF_RESULT_FILE_NAME
    VALIDATION_FILE_NAME = 'one_class_mf_validation.csv'
    TRAIN_FILE_NAME = 'one_class_mf_train.csv'
    MF_WEIGHT_DIR = Path('one_class_mf_weights_sgd')
    RESULT_DIR = Path(r'results/one_class')

# input configuration as recieved in the assignment
TRAIN_PATH = 'data/Train.csv'
RANDOM_TEST_PATH = 'data/RandomTest.csv'
RANDOM_TEST_COL_NAME1 = 'Item1'
RANDOM_TEST_COL_NAME2 = 'Item2'
USERS_COL_NAME = 'UserID'
ITEMS_COL_NAME = 'ItemID'

# internal column names
USER_COL = 'user'
ITEM_COL = 'item'
RANK_COL = 'rank'
positive_col = 'positive'
negative_col = 'negative'
# TEST_PATH = 'data/Test.csv'

# RATING_COL_NAME = 'Ratings_Rating'
# CHECKPOINT_NAME = f"./checkpoint_{model_name}.pkl"
# HYPER_PARAM_FILE_NAME = f"HyperParamResult_{model_name}.pkl"
