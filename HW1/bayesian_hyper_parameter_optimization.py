import numpy as np
import pandas as pd
import skopt

from skopt import gp_minimize
from skopt.plots import plot_convergence
from skopt.callbacks import CheckpointSaver

from HW1.MatrixFactorizationModelSGD import MatrixFactorizationWithBiasesSGD
from HW1.MartixFactorizationModelALS import MatrixFactorizationWithBiasesALS
from HW1.config import TRAIN_PATH, VALIDATION_PATH, HYPER_PARAM_SEARCH, ALS_SPACE, SGD, SGD_SPACE, SGD_CONFIG, \
    ALS_CONFIG, HYPER_PARAM_SEARCH_N_ITER, SEED
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


def run_exp(model):
    if HYPER_PARAM_SEARCH:
        checkpoint_saver = CheckpointSaver("./checkpoint.pkl")
        res_gp = gp_minimize(objective, space, n_calls=HYPER_PARAM_SEARCH_N_ITER, random_state=SEED,
                             callback=[checkpoint_saver])
        print(res_gp.x)
        print(res_gp.fun)
        skopt.dump(res_gp, 'SGD_Result.pkl')
        plot_convergence(res_gp)
    else:
        model.fit(train, validation, user_map, item_map)


if __name__ == '__main__':
    train, validation = pd.read_csv(TRAIN_PATH), pd.read_csv(VALIDATION_PATH)
    train, validation, user_map, item_map = preprocess_for_mf(train, validation)
    n_users, n_items = len(user_map), len(item_map)
    mf = get_mf(n_users, n_items)
    run_exp(mf)