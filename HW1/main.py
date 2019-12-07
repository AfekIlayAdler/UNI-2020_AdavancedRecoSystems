import numpy as np
import pandas as pd
import skopt
import matplotlib.pyplot as plt

from skopt import gp_minimize
from skopt.plots import plot_convergence

from skopt.callbacks import CheckpointSaver

from HW1.MatrixFactorizationModelSGD import MatrixFactorizationWithBiasesSGD
from HW1.MartixFactorizationModelALS import MatrixFactorizationWithBiasesALS
from HW1.config import TRAIN_PATH, VALIDATION_PATH, HYPER_PARAM_SEARCH
from HW1.utils import preprocess_for_mf, Config

SPACE = [skopt.space.Real(0.005, 0.012, name='l2_users', prior='uniform'),
         skopt.space.Real(0.005, 0.012, name='l2_items', prior='uniform'),
         skopt.space.Real(0.005, 0.012, name='l2_users_bias', prior='uniform'),
         skopt.space.Real(0.005, 0.012, name='l2_items_bias', prior='uniform'),
         skopt.space.Real(0.8, 0.95, name='beta', prior='uniform'),
         skopt.space.Categorical([16, 24, 32], name='h_len')]


@skopt.utils.use_named_args(SPACE)
def objective(**params):
    mf.set_params(**params)
    print({i: np.round(v, 3) for i, v in mf.__dict__.items() if i in params.keys()})
    return mf.fit(train, validation, user_map, item_map)


train, validation = pd.read_csv(TRAIN_PATH), pd.read_csv(VALIDATION_PATH)
train, validation, user_map, item_map = preprocess_for_mf(train, validation)
n_users, n_items = len(user_map), len(item_map)
results = []
config = Config(
    print_metrics=True,
    hidden_dimension=18, lr=0.01,
    l2_users=0.01,
    l2_items=0.01,
    l2_users_bias=0.001,
    l2_items_bias=0.001,
    epochs=50,
    bias_epochs=5,
    n_users=n_users, n_items=n_items, seed=1)

mf = MatrixFactorizationWithBiasesSGD(config)
if HYPER_PARAM_SEARCH:
    checkpoint_saver = CheckpointSaver("./checkpoint.pkl")
    res_gp = gp_minimize(objective, SPACE, n_calls=50, random_state=0, callback=[checkpoint_saver])
    print(res_gp.x)
    print(res_gp.fun)
    skopt.dump(res_gp, 'SGD_Result.pkl')
    plot_convergence(res_gp)
else:
    mf.fit(train, validation, user_map, item_map)
# results.append(mf.get_results())
# pd.concat(results, axis=1).to_csv('elbow.csv')
# if __name__ == "__main__":
#     main()
