from pathlib import Path

from optimization_objects import Config

NEGATIVE_SAMPLER_TYPE = 'popularity'   # 'uniform' ,'popularity'
VALIDATION_CREATOR_SAMPLER_TYPE = 'popularity'   # 'uniform' ,'popularity'
PREDICT_ON = 'popularity' # "random" 'popularity'
assert VALIDATION_CREATOR_SAMPLER_TYPE == PREDICT_ON, 'predict on wrong test'
FIT_ON_TRAIN_VALIDATION = True
# Save and load
MF_SAVE_TRAIN_VALIDATION = True
MF_LOAD_TRAIN_VALIDATION = False
#change to true
LOAD_NEGATIVE = False

K_LIST_FOR_PRECISION_AT_K = [1, 10, 50]

SEED = 5
CONFIG = Config(
    lr=0.05,
    print_metrics=True,
    beta=0.9,
    hidden_dimension=32,
    l2_users=0.01,
    l2_items=0.01,
    l2_items_bias=0.001,
    epochs=35,
    bias_epochs=2,
    seed=SEED,
    negative_sampler_type=NEGATIVE_SAMPLER_TYPE,
    validation_creator_sampler_type=VALIDATION_CREATOR_SAMPLER_TYPE)

# results
#NEGATIVE_SAMPLES_FILE_NAME = F"negative_samples_{NEGATIVE_SAMPLER_TYPE}"
NEGATIVE_SAMPLES_FILE_NAME = F"all_negative_samples_{NEGATIVE_SAMPLER_TYPE}"
RESULT_FILE_NAME = F"validation_results_negative_sampler_{NEGATIVE_SAMPLER_TYPE}_validation_type_{VALIDATION_CREATOR_SAMPLER_TYPE}.csv"
TRAIN_FILE_NAME = F"train_validation_type_{VALIDATION_CREATOR_SAMPLER_TYPE}.csv"
VALIDATION_FILE_NAME = F"validation_validation_type_{VALIDATION_CREATOR_SAMPLER_TYPE}.csv"
PREDICTION_FILE_NAME = F"prediction_negative_sampler_{NEGATIVE_SAMPLER_TYPE}_validation_type_{VALIDATION_CREATOR_SAMPLER_TYPE}.csv"

INTERNAL_DATA_DIR = Path('train_internal_data')
RESULT_DIR = Path(r'results')

# input configuration as received in the assignment
TRAIN_PATH = 'data/Train.csv'

if PREDICT_ON == "popularity":
    TEST_PATH = 'data/RandomTest.csv'
else:
    TEST_PATH = 'data/PopularityTest.csv'

N_NEGATIVE_SAMPLES_SETS = 50

RANDOM_TEST_COL_NAME1 = 'Item1'
RANDOM_TEST_COL_NAME2 = 'Item2'
USERS_COL_NAME = 'UserID'
ITEMS_COL_NAME = 'ItemID'

# internal column names
USER_COL = 'user'
ITEM_COL = 'item'
RANK_COL = 'rank'
POSITIVE_COL = 'positive'
NEGATIVE_COL = 'negative'
