import skopt

from HW1.optimization_objects import Config

# sgd or als
SGD = True
# HYPER_PARAM_SEARCH or manual config
HYPER_PARAM_SEARCH = True
HYPER_PARAM_SEARCH_N_ITER = 50
SEED = 3

# hyper parameter tuning
SGD_SPACE = [skopt.space.Real(0.005, 0.03, name='lr', prior='uniform'),
             skopt.space.Real(0.001, 0.012, name='l2_users', prior='uniform'),
             skopt.space.Real(0.001, 0.012, name='l2_items', prior='uniform'),
             skopt.space.Real(0.001, 0.012, name='l2_users_bias', prior='uniform'),
             skopt.space.Real(0.001, 0.012, name='l2_items_bias', prior='uniform'),
             skopt.space.Categorical([16, 18, 20, 24, 28, 32], name='h_len')]

SGD_CONFIG = Config(
    print_metrics=True,
    beta=0.9,
    hidden_dimension=18,
    lr=0.025,
    l2_users=0.01,
    l2_items=0.01,
    l2_users_bias=0.001,
    l2_items_bias=0.001,
    epochs=30,
    bias_epochs=5,
    seed=SEED)

ALS_SPACE = [skopt.space.Real(0.005, 0.012, name='l2_users', prior='uniform'),
             skopt.space.Real(0.005, 0.012, name='l2_items', prior='uniform'),
             skopt.space.Real(0.005, 0.012, name='l2_users_bias', prior='uniform'),
             skopt.space.Real(0.005, 0.012, name='l2_items_bias', prior='uniform'),
             skopt.space.Categorical([16, 24, 32], name='h_len')]

ALS_CONFIG = Config(
    print_metrics=True,
    hidden_dimension=18,
    l2_users=0.01,
    l2_items=0.01,
    l2_users_bias=0.001,
    l2_items_bias=0.001,
    epochs=2,
    bias_epochs=2,
    seed=SEED)

TRAIN_PATH = 'data/Train.csv'
VALIDATION_PATH = 'data/Validation.csv'
USERS_COL_NAME = 'User_ID_Alias'
ITEMS_COL_NAME = 'Movie_ID_Alias'
RATING_COL_NAME = 'Ratings_Rating'
USER_COL = 'user'
ITEM_COL = 'item'
model_name = 'sgd' if SGD else 'als'
CHECKPOINT_NAME = f"./checkpoint_{model_name}.pkl"
HYPER_PARAM_FILE_NAME = f"HyperParamResult_{model_name}.pkl"
