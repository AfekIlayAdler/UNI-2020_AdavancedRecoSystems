import numpy as np
import pandas as pd


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
            result = "{:<32}".format(F"  {key} : {val}")
            temp += result
        print(temp)

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def fit(self, train: pd.DataFrame, user_map: dict, item_map: dict, validation: np.array = None):
        pass

    def l2_loss(self):
        loss = 0
        parameters = [self.item_biases, self.U, self.V]
        regularizations = [self.l2_items_bias, self.l2_users, self.l2_items]
        for i in range(len(parameters)):
            loss += regularizations[i] * np.sum(np.square(parameters[i]))
        return loss

    def sigmoid_inner_scalar(self, user, item):
        return self.item_biases[item] + self.U[user, :].dot(self.V[item, :].T)

