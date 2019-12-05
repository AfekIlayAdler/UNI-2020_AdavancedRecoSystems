import numpy as np
import pandas as pd


class MatrixFactorizationWithBiases:
    def __init__(self, seed, hidden_dimension):
        self.h_len = hidden_dimension
        self.results = {}
        np.random.seed(seed)

    def get_results(self):
        return pd.DataFrame.from_dict(self.results)

    def record(self, epoch, **kwargs):
        print(f"epoch # {epoch} : \n")
        for key, value in kwargs.items():
            key = f"{key}_{self.h_len}"
            if not self.results.get(key):
                self.results[key] = []
            self.results[key].append(value)
            print(f"{key} : {np.round(value, 5)}")

    def fit(self, train: pd.DataFrame, validation: pd.DataFrame, user_map: dict, item_map: dict):
        pass

    def predict(self, user, item):
        """
        predict on user and item with their original ids not internal ids
        """
        user = user.get(self.user_map, None)
        item = item.get(self.item_map, None)
        if user:
            if item:
                prediction = self.predict_on_pair(user, item)
            else:
                prediction = self.predict_on_existing_user_new_item(user)
        else:
            if item:
                prediction = self.predict_on_new_user_existing_item(item)
            else:
                prediction = self.predict_on_new_user_new_item()
        return np.clip(prediction, 1, 5)

    def calc_loss(self, x):
        loss = 0
        parameters = [self.user_biases, self.item_biases, self.U, self.V]
        regularizations = [self.l2_users_bias, self.l2_items_bias, self.l2_users, self.l2_items]
        for i in range(len(parameters)):
            loss += regularizations[i] * np.sum(np.square(parameters[i]))
        return loss + self.prediction_error(x)

    def r2(self, x):
        denominator = 0
        nominator = 0
        for row in x:
            user, item, rating = row
            denominator += np.square(self.predict_on_pair(user, item)-self.global_bias)
            nominator += np.square(rating - self.predict_on_pair(user, item))
        return float(nominator)/denominator


    def mse(self, x):
        e = 0
        for row in x:
            user, item, rating = row
            e += np.square(rating - self.predict_on_pair(user, item))
        return e / x.shape[0]


    def rmse(self, x):
        e = 0
        for row in x:
            user, item, rating = row
            e += np.square(rating - self.predict_on_pair(user, item))
        return np.sqrt(e / x.shape[0])

    def mae(self, x):
        e = 0
        for row in x:
            user, item, rating = row
            e += np.abs(rating - self.predict_on_pair(user, item))
        return e / x.shape[0]

    def prediction_error(self, x, measure_function="rmse"):
        if measure_function == "rmse":
            return self.rmse(x)
        elif measure_function == "mae":
            return self.mae(x)

    def predict_on_pair(self, user, item):
        return self.global_bias + self.user_biases[user] + self.item_biases[item] \
               + self.U[user, :].dot(self.V[item, :].T)

    def predict_on_new_user_existing_item(self, item):
        return self.global_bias + self.item_biases[item]

    def predict_on_existing_user_new_item(self, user):
        return self.global_bias + self.user_biases[user]

    def predict_on_new_user_new_item(self):
        return self.global_bias
