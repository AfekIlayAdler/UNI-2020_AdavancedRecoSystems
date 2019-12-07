import numpy as np

from HW1.optimization_objects import LearningRateScheduler, EarlyStopping
from HW1.matrix_factorization_abstract import MatrixFactorizationWithBiases
from HW1.momentum_wrapper import MomentumWrapper1D, MomentumWrapper2D


class MatrixFactorizationWithBiasesSGD(MatrixFactorizationWithBiases):
    # initialization of model's parameters
    def __init__(self, config):
        super().__init__(config.seed, config.hidden_dimension, config.print_metrics)
        self.n_users = config.n_users
        self.n_items = config.n_items
        self.lr = LearningRateScheduler(config.lr)
        self.early_stopping = EarlyStopping(2)
        self.l2_users = config.l2_users
        self.l2_items = config.l2_items
        self.l2_users_bias = config.l2_users_bias
        self.l2_items_bias = config.l2_items_bias
        self.epochs = config.epochs
        self.number_bias_epochs = config.bias_epochs
        none_args = ['global_bias', 'user_biases', 'item_biases', 'U', 'V', 'users_h_gradient', 'items_h_gradient',
                     'user_map', 'item_map', 'user_biases_gradient', 'item_biases_gradient']
        for arg in none_args:
            setattr(self, arg, None)
        self.beta = 0.9
        self.results = {}

    # initialization of model's weights
    def weight_init(self, user_map, item_map, global_bias):
        self.global_bias = global_bias
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

    def fit(self, train, validation, user_map: dict, item_map: dict):
        """data columns: [user id,movie_id,rating in 1-5]"""
        train, validation = train.values, validation.values
        self.weight_init(user_map, item_map, np.mean(train[:, 2]))
        for epoch in range(1, self.epochs + 1):
            np.random.shuffle(train)
            self.run_epoch(train, epoch)
            # calculate train/validation error and loss
            validation_error = self.prediction_error(validation)
            self.record(epoch, train_accuracy=self.prediction_error(train),
                        test_accuracy=validation_error,
                        train_loss=self.calc_loss(train), test_loss=self.calc_loss(validation))
            if self.early_stopping.stop(self,epoch, validation_error):
                break
        print(f"validation_error: {validation_error}")
        return validation_error

    def run_epoch(self, data, epoch):
        lr = self.lr.update(epoch)
        for row in data:
            user, item, rating = row
            prediction = self.predict_on_pair(user, item)
            error = rating - prediction
            u_b_gradient = (error - self.l2_users_bias * self.user_biases[user])
            i_b_gradient = (error - self.l2_items_bias * self.item_biases[item])
            self.user_biases[user] += lr * self.user_biases_gradient.get(u_b_gradient, user)
            self.item_biases[item] += lr * self.item_biases_gradient.get(i_b_gradient, item)
            if epoch > self.number_bias_epochs:
                u_grad = (error * self.V[item, :] - self.l2_users * self.U[user, :])
                v_grad = (error * self.U[user, :] - self.l2_items * self.V[item, :])
                self.U[user, :] += lr * self.users_h_gradient.get(u_grad, user)
                self.V[item, :] += lr * self.items_h_gradient.get(v_grad, item)
