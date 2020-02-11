import numpy as np
import pandas as pd
from scipy.special import expit

from HW2.config import USER_COL, K_LIST_FOR_PRECISION_AT_K, POSITIVE_COL, ITEM_COL, SEED, TEST_PATH, \
    PREDICTION_FILE_NAME, RESULT_DIR
from matrix_factorization_abstract import MatrixFactorizationWithBiases
from momentum_wrapper import MomentumWrapper1D, MomentumWrapper2D
from nagative_sampler import NegativeSampler
from optimization_objects import LearningRateScheduler, SgdEarlyStopping
from utils import sigmoid, get_item_probabilities


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
        self.negative_sampler_type = config.negative_sampler_type
        np.random.seed(SEED)

    # initialization of model's weights
    def weight_init(self, user_map, item_map):
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
        self.negative_sampler = NegativeSampler(get_item_probabilities(train), method=self.negative_sampler_type)
        self.negative_sampler.create_negative_samples(train)
        self.early_stopping = SgdEarlyStopping()
        self.lr = LearningRateScheduler(self.lr)
        self.weight_init(user_map, item_map)
        validation_error = None
        for epoch in range(1, self.epochs + 1):
            train_with_negative_samples = self.negative_sampler.get(epoch)
            np.random.shuffle(train_with_negative_samples)
            self.run_epoch(train_with_negative_samples, epoch)
            train_percent_right, train_log_likelihood = self.percent_right_and_log_likelihood(
                train_with_negative_samples)
            train_objective = train_log_likelihood - self.l2_loss()
            convergence_params = {'train_objective': train_objective, 'train_percent_right': train_percent_right}
            if validation is not None:
                validation_percent_right, validation_log_likelihood = self.percent_right_and_log_likelihood(
                    validation.values)
                if self.early_stopping.stop(self, epoch, validation_percent_right):
                    break
                convergence_params.update({'validation_objective': validation_log_likelihood,
                                           'validation_percent_right': validation_percent_right})
                convergence_params.update(self.calculate_precision_at_k(train, validation))
                self.record(epoch, **convergence_params)
        return validation_error

    def run_epoch(self, data, epoch):
        lr = self.lr.update(epoch)
        data = data.astype(int)
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

    def calculate_precision_at_k(self, train, val):
        unique_items = set([i for i in range(self.n_items)])
        ranks_all_items = np.zeros(self.n_users)
        ranks_items_user_did_not_rank = np.zeros(self.n_users)
        for user in range(self.n_users):
            user_validation_item = val[val[USER_COL] == user][POSITIVE_COL].values[0]
            user_unique_items = set(train[train[USER_COL] == user][ITEM_COL])
            user_items_did_not_rank = list(unique_items.difference(user_unique_items))
            relative_ranks = pd.Series(expit(self.V.dot(self.U[user, :]) + self.item_biases))
            ranks_items_user_did_not_rank[user] = relative_ranks.take(user_items_did_not_rank).rank(ascending=False)[
                user_validation_item]
            ranks_all_items[user] = relative_ranks.rank(ascending=False)[user_validation_item]
        results_dict = {}
        for k in K_LIST_FOR_PRECISION_AT_K:
            results_dict[F"precision_at_{k}_all_items"] = np.sum(ranks_all_items <= k) / self.n_users
            results_dict[F"precision_at_{k}_items_user_did_not_rank"] = np.sum(
                ranks_items_user_did_not_rank <= k) / self.n_users
        results_dict[F"mpr_all_items"] = np.mean(ranks_all_items / self.n_items)
        return results_dict

    def predict_likelihood(self, user, item):
        return sigmoid(self.sigmoid_inner_scalar(user, item))

    def percent_right_and_log_likelihood(self, x):
        x = x.astype(int)
        log_likelihood = 0
        counter = 0
        for row in x:
            user, item_positive, item_negative = row
            prediction = sigmoid(self.sigmoid_inner_scalar_pair(int(user), int(item_positive), int(item_negative)))
            counter += (prediction > 0.5)
            log_likelihood += np.log(prediction)
        percent_right = counter / x.shape[0]
        return percent_right, log_likelihood

    def predict_on_test_set(self):
        test = pd.read_csv(TEST_PATH)
        predictions = []
        for _, row in test.iterrows():
            user, item1, item2 = self.user_map.get(row['UserID']), self.item_map.get(
                row['Item1'], -1), self.item_map.get(row['Item2'], -1)
            # if item2 is more likely then item1 than 1
            # if the item does not exist we replace it with the mean
            if item1 + item2 == -2:
                # choose random
                prediction = np.random.random_sample()
            if item1 == -1:
                print('item exist in test but not in train')
                prediction = sigmoid(self.item_biases[item2] - np.mean(self.item_biases) + \
                                     self.U[user, :].dot(self.V[item2, :].T - np.mean(self.V, axis=0)))
            elif item2 == -1:
                print('item exist in test but not in train')
                prediction = sigmoid(np.mean(self.item_biases) - self.item_biases[item1] + \
                                      self.U[user, :].dot(np.mean(self.V, axis=0) - self.V[item1, :].T))
            else:
                prediction = sigmoid(self.sigmoid_inner_scalar_pair(user, item2, item1))
            predictions.append((prediction > 0.5) * 1)
        test['bitClassification'] = predictions
        test.to_csv(RESULT_DIR / PREDICTION_FILE_NAME, index=False)
