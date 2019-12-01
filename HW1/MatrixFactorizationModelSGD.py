import numpy as np

from HW1.matrix_factorization_abstract import MatrixFactorizationWithBiases
from HW1.momentum_wrapper import MomentumWrapper1D, MomentumWrapper2D


class MatrixFactorizationWithBiasesSGD(MatrixFactorizationWithBiases):
    def __init__(self, config):
        super().__init__(config.seed, config.hidden_dimension)
        self.n_users = config.n_users
        self.n_items = config.n_items
        self.lr = config.lr
        self.l2_users = config.l2_users
        self.l2_items = config.l2_items
        self.l2_users_bias = config.l2_users_bias
        self.l2_items_bias = config.l2_items_bias
        self.epochs = config.epochs
        self.number_bias_epochs = config.bias_epochs
        self.global_bias = None
        self.user_biases = None
        self.item_biases = None
        self.U = None  # users matrix
        self.V = None  # items matrix
        self.momentum = True
        self.users_h_gradient = None  # for momentum
        self.items_h_gradient = None  # for momentum
        self.user_map = None
        self.item_map = None
        self.user_biases_gradient = None
        self.item_biases_gradient = None
        self.beta = 0.7
        self.results = {}
        # TODO understand if we need self.batch_size

    def weight_init(self, user_map, item_map):
        self.user_map, self.item_map = user_map, item_map
        # TODO understand if we can get a better initialization
        self.U = np.random.normal(scale=1. / self.h_len, size=(self.n_users, self.h_len))
        self.V = np.random.normal(scale=1. / self.h_len, size=(self.n_items, self.h_len))
        self.users_h_gradient = MomentumWrapper2D(self.n_users, self.h_len, self.beta)
        self.items_h_gradient = MomentumWrapper2D(self.n_items, self.h_len, self.beta)
        # Initialize the biases
        self.user_biases = np.zeros(self.n_users)
        self.item_biases = np.zeros(self.n_items)
        self.user_biases_gradient = MomentumWrapper1D(self.n_users, self.beta)
        self.item_biases_gradient = MomentumWrapper1D(self.n_items, self.beta)

    def fit(self, train: np.array, validation: np.array, user_map: dict, item_map: dict):
        """data columns: [user id,movie_id,rating in 1-5]"""
        self.weight_init(user_map, item_map)
        self.global_bias = np.mean(train[:, 2])
        for epoch in range(1, self.epochs + 1):
            np.random.shuffle(train)
            self.run_epoch(train, epoch)
            self.record(epoch, train_accuracy=self.prediction_loss(train),
                        test_accuracy=self.prediction_loss(validation),
                        train_loss=self.calc_loss(train), test_loss=self.calc_loss(validation))
            if epoch == 10:
                self.lr *= 0.1

    def prediction_loss(self, x, rmse=True):
        e = 0
        for row in x:
            user, item, rating = row
            e += np.square(rating - self.predict_on_pair(user, item))
        if rmse:
            return np.sqrt(e / x.shape[0])
        return e

    def calc_loss(self, x):
        loss = 0
        parameters = [self.user_biases, self.item_biases, self.U, self.V]
        regularizations = [self.l2_users_bias, self.l2_items_bias, self.l2_users, self.l2_items]
        for i in range(len(parameters)):
            loss += regularizations[i] * np.sum(np.square(parameters[i]))
        return loss + self.prediction_loss(x, rmse=False)

    def run_epoch(self, data, epoch):
        for row in data:
            user, item, rating = row
            prediction = self.predict_on_pair(user, item)
            error = rating - prediction
            u_b_gradient = (error - self.l2_users_bias * self.user_biases[user])
            i_b_gradient = (error - self.l2_items_bias * self.item_biases[item])
            self.user_biases[user] += self.lr * self.user_biases_gradient.get(u_b_gradient, user)
            self.item_biases[item] += self.lr * self.item_biases_gradient.get(i_b_gradient, item)
            if epoch > self.number_bias_epochs:
                u_grad = (error * self.V[item, :] - self.l2_users * self.U[user, :])
                v_grad = (error * self.U[user, :] - self.l2_items * self.V[item, :])
                self.U[user, :] += self.lr * self.users_h_gradient.get(u_grad, user)
                self.V[item, :] += self.lr * self.items_h_gradient.get(v_grad, item)

