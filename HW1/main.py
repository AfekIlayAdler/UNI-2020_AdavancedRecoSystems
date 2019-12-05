import pandas as pd
import pickle
from HW1.MatrixFactorizationModelSGD import MatrixFactorizationWithBiasesSGD
from HW1.MartixFactorizationModelALS import MatrixFactorizationWithBiasesALS
from HW1.config import TRAIN_PATH, VALIDATION_PATH
from HW1.utils import preprocess_for_mf, Config
import numpy as np


def find_optimal_architecture(algo):
    latent_factors = [5, 10, 16, 20, 40, 80]
    regularizations = [0.01, 0.1, 1., 10., 100.]
    iter_array = [10, 20, 50, 100]
    if algo=="SGD":
        learning_rates = [1e-5, 1e-4, 1e-3, 1e-2]
    else:
        learning_rates = [0.01]

    best_params = {}
    best_params['n_factors'] = latent_factors[0]
    best_params['reg'] = regularizations[0]
    best_params['n_iter'] = 0
    best_params['train'] = np.inf
    best_params['test'] = np.inf
    best_params['model'] = None

    for fact in latent_factors:
        print ('Factors: {}'.format(fact))
        for reg in regularizations:
            print ('Regularization: {}'.format(reg))
            for rate in learning_rates:
                mf, results = main(hidden_dimension=fact, regularizations=reg, epochs=2, lr=rate, algo=algo)
                min_idx = results["test_loss_{}".format(fact)].idxmin()
                if results["test_loss_{}".format(fact)][min_idx] < best_params['test']:
                    best_params['n_factors'] = fact
                    best_params['reg'] = reg
                    best_params['n_iter'] = min_idx
                    best_params['train'] = results["train_loss_{}".format(fact)][min_idx]
                    best_params['test'] = results["test_loss_{}".format(fact)][min_idx]
                    best_params['model'] = mf
    print ('optimal hyperparameters')
    print (pd.Series(best_params))

    with open('best_model.pkl', 'wb') as output:
        pickle.dump(best_params, output)

def main(**kwargs):
    train, validation = pd.read_csv(TRAIN_PATH), pd.read_csv(VALIDATION_PATH)
    train, validation, user_map, item_map = preprocess_for_mf(train, validation)
    n_users = len(user_map)
    n_items = len(item_map)
    results = []
    config = Config(
        hidden_dimension=kwargs["hidden_dimension"], lr=kwargs["lr"],
        l2_users=kwargs["regularizations"],
        l2_items=kwargs["regularizations"],
        l2_users_bias=kwargs["regularizations"],
        l2_items_bias=kwargs["regularizations"],
        epochs=kwargs["epochs"],
        bias_epochs=1,
        n_users=n_users, n_items=n_items, seed=1)
    if kwargs["algo"]=="SGD":
        mf = MatrixFactorizationWithBiasesALS(config)
        mf.fit(train, validation, user_map, item_map)
    else:
        mf = MatrixFactorizationWithBiasesALS(config)
        mf.fit(train, validation, user_map, item_map)
    results.append(mf.get_results())
    pd.concat(results, axis=1).to_csv('elbow.csv')
    return mf, mf.get_results()

if __name__ == "__main__":
    #main(hidden_dimension=16, regularizations=0.002, epochs=2, lr=0.01)
    find_optimal_architecture("ALS")
