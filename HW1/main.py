import numpy as np
import pandas as pd
import skopt

from skopt import gp_minimize
from skopt.plots import plot_convergence
#from skopt.callbacks import CheckpointSaver

from HW1.MatrixFactorizationModelSGD import MatrixFactorizationWithBiasesSGD
from HW1.MartixFactorizationModelALS import MatrixFactorizationWithBiasesALS
from HW1.config import TRAIN_PATH, VALIDATION_PATH, HYPER_PARAM_SEARCH, ALS_SPACE, SGD, SGD_SPACE, SGD_CONFIG, \
    ALS_CONFIG, HYPER_PARAM_SEARCH_N_ITER, SEED, CHECKPOINT_NAME, HYPER_PARAM_FILE_NAME, x0, y0
from HW1.utils import preprocess_for_mf

space = SGD_SPACE if SGD else ALS_SPACE


def get_mf(users_len, items_len):
    conf = SGD_CONFIG if SGD else ALS_CONFIG
    conf.add_attributes(n_users=users_len, n_items=items_len)
    if SGD:
        return MatrixFactorizationWithBiasesSGD(conf)
    return MatrixFactorizationWithBiasesALS(conf)


@skopt.utils.use_named_args(space)
def objective(**params):
    mf.set_params(**params)
    print({i: np.round(v, 3) for i, v in mf.__dict__.items() if i in params.keys()})
    return mf.fit(train, validation, user_map, item_map)


def run_exp(model,train, user_map, item_map, validation=None, last_run=False):
    if HYPER_PARAM_SEARCH:
        #checkpoint_saver = CheckpointSaver(CHECKPOINT_NAME)
        #callback=[checkpoint_saver]
        res_gp = gp_minimize(objective, space, n_calls=HYPER_PARAM_SEARCH_N_ITER, random_state=SEED,
                              x0=x0, y0=y0)
        print(res_gp.x)
        print(res_gp.fun)
        skopt.dump(res_gp, HYPER_PARAM_FILE_NAME,store_objective=False)
        plot_convergence(res_gp)
    else:
        if last_run:
            model.fit_all(train, user_map, item_map)
        else:
            model.fit(train, validation, user_map, item_map)
    return model

if __name__ == '__main__':
    train, validation = pd.read_csv(TRAIN_PATH), pd.read_csv(VALIDATION_PATH)
    train_all = pd.concat([train, validation], ignore_index=True)
    train, validation, user_map, item_map = preprocess_for_mf(train, validation)
    n_users, n_items = len(user_map), len(item_map)
    mf = get_mf(n_users, n_items)
    trained_model = run_exp(mf, train, user_map, item_map, validation)

    # Final Run on all of the train data
    train_all, user_map, item_map = preprocess_for_mf(train_all)
    n_users, n_items = len(user_map), len(item_map)
    trained_model.n_users = n_users
    trained_model.n_items = n_items
    FinalModel = run_exp(trained_model, train, user_map, item_map, validation=None, last_run=True)
