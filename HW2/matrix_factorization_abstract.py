import numpy as np
import pandas as pd
from scipy.special import expit as logit

from HW2.utils import sigmoid


class MatrixFactorizationWithBiases:
    def __init__(self, seed, hidden_dimension, print_metrics=True):
        self.h_len = hidden_dimension
        self.results = {}
        np.random.seed(seed)
        self.print_metrics = print_metrics
        self.user_map = None
        self.item_map = None
        self.user_biases = None
        self.item_biases = None
        self.U = None
        self.V = None
        self.l2_users_bias = None
        self.l2_items_bias = None
        self.l2_users = None
        self.l2_items = None
        self.global_bias = None

    def get_results(self):
        return pd.DataFrame.from_dict(self.results)

    def record(self, epoch, **kwargs):
        epoch = "{:02d}".format(epoch)
        temp = f"| epoch   # {epoch} :"
        for key, value in kwargs.items():
            key = f"{key}"
            if not self.results.get(key):
                self.results[key] = []
            self.results[key].append(value)
            val = '{:.4}'.format(value)
            temp += f"{key} : {val}      |       "
        print(temp)

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def fit(self, train: pd.DataFrame, user_map: dict, item_map: dict, validation: np.array = None):
        pass

    def predict(self, user, item):
        """
        predict on user and item with their original ids not internal ids
        """
        user = self.user_map.get(user, None)
        item = self.item_map.get(item, None)
        # TODO: remove this check
        if (user is None) or (item is None):
            print('item or user is none')
        if user:
            if item:
                prediction = self.predict_on_pair(user, item)
            else:
                prediction = self.global_bias + self.user_biases[user]
        else:
            if item:
                prediction = self.global_bias + self.item_biases[item]
            else:
                prediction = self.global_bias
        return np.clip(prediction, 1, 5)

    def calc_loss(self, x):
        loss = 0
        parameters = [self.user_biases, self.item_biases, self.U, self.V]
        regularizations = [self.l2_users_bias, self.l2_items_bias, self.l2_users, self.l2_items]
        for i in range(len(parameters)):
            loss += regularizations[i] * np.sum(np.square(parameters[i]))
        return loss

    def prediction_error(self, x):
        nll = 0
        for row in x:
            user, item, rating = row
            error = rating * np.log(self.predict_on_pair(user, item)) + (1 - rating) * np.log(
                1 - self.predict_on_pair(user, item))
            nll += error
        return -1 *( nll / x.shape[0])

    def predict_on_pair(self, user, item):
        return sigmoid(self.global_bias + self.user_biases[user] + self.item_biases[item] \
                       + self.U[user, :].dot(self.V[item, :].T))
