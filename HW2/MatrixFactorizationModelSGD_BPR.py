import numpy as np
from IPython.core.display import display, HTML

from nagative_sampler import NegativeSampler
from optimization_objects import LearningRateScheduler, SgdEarlyStopping
from matrix_factorization_abstract import MatrixFactorizationWithBiases
from momentum_wrapper import MomentumWrapper1D, MomentumWrapper2D
from HW2.config import ITEM_COL, USER_COL, positive_col, negative_col

from utils import sigmoid, get_item_probabilities
import pandas as pd


class BPRMatrixFactorizationWithBiasesSGD(MatrixFactorizationWithBiases):
    # initialization of model's parameters
    def __init__(self, config):
        super().__init__(config.seed, config.hidden_dimension, config.print_metrics)
        self.n_users = config.n_users
        self.n_items = config.n_items
        self.lr = config.lr
        self.negative_sampler = None
        self.early_stopping = None
        self.l2_users = config.l2_users
        self.l2_items = config.l2_items
        self.l2_items_bias = config.l2_items_bias
        self.epochs = config.epochs
        self.number_bias_epochs = config.bias_epochs
        self.beta = config.beta
        self.results = {}
        self.users_h_gradient = None
        self.items_h_gradient = None
        self.user_biases_gradient = None
        self.item_biases_gradient = None

    # initialization of model's weights
    def weight_init(self, user_map, item_map, global_bias):
        self.user_map, self.item_map = user_map, item_map
        self.U = np.random.normal(scale=0.2 / self.h_len, size=(self.n_users, self.h_len))
        self.V = np.random.normal(scale=0.2 / self.h_len, size=(self.n_items, self.h_len))
        self.users_h_gradient = MomentumWrapper2D(self.n_users, self.h_len, self.beta)
        self.items_h_gradient = MomentumWrapper2D(self.n_items, self.h_len, self.beta)
        # Initialize the biases
        self.item_biases = np.zeros(self.n_items)
        self.item_biases_gradient = MomentumWrapper1D(self.n_items, self.beta)

    def fit(self, train, user_map: dict, item_map: dict, validation=None):
        """data columns: [user id,movie_id,like or not {0,1}]"""
        self.negative_sampler = NegativeSampler(get_item_probabilities(train), method='popularity')
        self.early_stopping = SgdEarlyStopping()
        self.lr = LearningRateScheduler(self.lr)
        self.weight_init(user_map, item_map, len(train) / len(user_map) * len(item_map))
        validation_error = None
        for epoch in range(1, self.epochs + 1):
            train_with_negative_samples = self.negative_sampler.get(train, epoch)
            np.random.shuffle(train_with_negative_samples)
            self.run_epoch(train_with_negative_samples, epoch)
            train_percent_right, train_log_likelihood = self.percent_right_and_log_likelihood(train_with_negative_samples)
            train_objective = train_log_likelihood - self.l2_loss()
            convergence_params = {'train_objective': train_objective, 'train_percent_right': train_percent_right}
            if validation is not None:
                validation_percent_right, validation_log_likelihood = self.percent_right_and_log_likelihood(validation.values)
                if self.early_stopping.stop(self, epoch, validation_log_likelihood):
                    break
                convergence_params.update({'validation_objective': validation_log_likelihood, 'validation_percent_right': validation_percent_right})
                # precision_at_k_dict = self.get_precision_at_k_dict()
                # convergence_params.update(precision_at_k_dict)
                # self.record(epoch, **convergence_params)
                print(pd.Series(convergence_params).to_frame().T.to_string())

        return validation_error

    def run_epoch(self, data, epoch):
        lr = self.lr.update(epoch)
        for row in data:
            user, item_positive, item_negative = row
            error = 1 - sigmoid(self.sigmoid_inner_scalar_pair(user, item_positive, item_negative))
            i_p_b_gradient = error - self.l2_items_bias * self.item_biases[item_positive]
            i_n_b_gradient = -error - self.l2_items_bias * self.item_biases[item_negative]
            self.item_biases[item_positive] += lr * self.item_biases_gradient.get(i_p_b_gradient, item_positive)
            self.item_biases[item_negative] += lr * self.item_biases_gradient.get(i_n_b_gradient, item_positive)
            if epoch > self.number_bias_epochs:
                u_grad = error * (self.V[item_positive, :] - self.V[item_negative, :]) - self.l2_users * self.U[user, :]
                i_p_grad = error * self.U[user, :] - self.l2_items * self.V[item_positive, :]
                i_n_grad = -1 * error * self.U[user, :] - self.l2_items * self.V[item_negative, :]
                self.U[user, :] += lr * self.users_h_gradient.get(u_grad, user)
                self.V[item_positive, :] += lr * self.items_h_gradient.get(i_p_grad, item_positive)
                self.V[item_negative, :] += lr * self.items_h_gradient.get(i_n_grad, item_negative)

    def sigmoid_inner_scalar_pair(self, user, item_positive, item_negative):
        return self.item_biases[item_positive] - self.item_biases[item_negative] + \
               self.U[user, :].dot(self.V[item_positive, :].T - self.V[item_negative, :].T)

    def calculate_precision_at_k(self):
        # TODO make it more sexy
        pass
        # unique_items = set(item_map.values())
        # unique_users_in_validation = val[USER_COL].unique()
        # results = []
        # for user in unique_users_in_validation:
        #     current_user_val = val[val[USER_COL] == user]
        #     user_unique_items = set(train[train[:, 0] == user][:, 1])
        #     user_items_did_not_rank = list(unique_items.difference(user_unique_items))
        #     likelihood = {}
        #     for did_not_rank in user_items_did_not_rank:
        #         likelihood[did_not_rank] = self.predict_likelihood(user, did_not_rank)
        #     likelihood = pd.DataFrame.from_dict(likelihood, orient='index', columns=['likelihood'])
        #     likelihood.sort_values(by=['likelihood'], inplace=True, ascending=False)
        #     likelihood = likelihood.reset_index().rename(columns={'index': 'item'})
        #     for row in current_user_val.values:
        #         user, item_positive, item_negative = row
        #         index_in_likelihood = likelihood[likelihood['item'] == item_positive].index[0]
        #         results.append(index_in_likelihood)
        # return results

    def predict_likelihood(self, user, item):
        return sigmoid(self.sigmoid_inner_scalar(user, item))

    def percent_right_and_log_likelihood(self, x):
        log_likelihood = 0
        counter = 0
        for row in x:
            user, item_positive, item_negative = row
            prediction = sigmoid(self.sigmoid_inner_scalar_pair(user, item_positive, item_negative))
            counter += (prediction > 0.5)
            log_likelihood += np.log(prediction)
        percent_right = counter / x.shape[0]
        return percent_right, log_likelihood
