import numpy as np

from HW2.matrix_factorization_abstract import MatrixFactorizationWithBiases
from HW2.momentum_wrapper import MomentumWrapper2D, MomentumWrapper1D
from HW2.optimization_objects import SgdEarlyStopping, LearningRateScheduler
from utils import sigmoid
from validation_creator import create_validation_two_columns


class OneClassMatrixFactorizationWithBiasesSGD(MatrixFactorizationWithBiases):
    # initialization of model's parameters
    def __init__(self, config, negative_sampler):
        # TODO remove global bias
        super().__init__(config.seed, config.hidden_dimension, config.print_metrics)
        self.negative_sampler = negative_sampler
        self.lr = config.lr
        self.early_stopping = None
        self.l2_users = config.l2_users
        self.l2_items = config.l2_items
        self.l2_users_bias = config.l2_users_bias
        self.l2_items_bias = config.l2_items_bias
        self.epochs = config.epochs
        self.number_bias_epochs = config.bias_epochs
        self.beta = config.beta
        self.results = {}
        self.n_users = config.n_users
        self.n_items = config.n_items
        self.users_h_gradient = None
        self.items_h_gradient = None
        self.user_biases_gradient = None
        self.item_biases_gradient = None

    # initialization of model's weights
    def weight_init(self):
        self.global_bias = 0
        self.U = np.random.normal(scale=0.2 / self.h_len, size=(self.n_users, self.h_len))
        self.V = np.random.normal(scale=0.2 / self.h_len, size=(self.n_items, self.h_len))
        self.users_h_gradient = MomentumWrapper2D(self.n_users, self.h_len, self.beta)
        self.items_h_gradient = MomentumWrapper2D(self.n_items, self.h_len, self.beta)
        # Initialize the biases
        self.user_biases = np.zeros(self.n_users)
        self.item_biases = np.zeros(self.n_items)
        self.user_biases_gradient = MomentumWrapper1D(self.n_users, self.beta)
        self.item_biases_gradient = MomentumWrapper1D(self.n_items, self.beta)

    def fit(self, train, user_map: dict, item_map: dict, validation=None):
        """data columns: [user id,movie_id,rating or not {0,1}]"""
        self.early_stopping = SgdEarlyStopping()
        self.lr = LearningRateScheduler(self.lr)
        self.weight_init()
        validation_error = None
        for epoch in range(1, self.epochs + 1):
            train_with_negative_samples = self.negative_sampler.get(train, epoch)
            np.random.shuffle(train_with_negative_samples)
            self.run_epoch(train_with_negative_samples, epoch)
            # calculate train/validation error and loss
            train_mean_NLL = self.prediction_error(train_with_negative_samples)
            train_loss = self.calc_loss() + train_mean_NLL
            convergence_params = {'train_MNLL': train_mean_NLL, 'train_loss': train_loss}
            if validation is not None:
                if epoch == 1:
                    validation, validation_choose_between_two = create_validation_two_columns(validation)
                validation_mean_NLL = self.prediction_error(validation)
                validation_loss = self.calc_loss() + validation_mean_NLL
                percent_right_choices = self.predict_which_item_more_likely(validation_choose_between_two)
                # TODO change later to percent_right_choices
                if self.early_stopping.stop(self, epoch, validation_mean_NLL):
                    break
                convergence_params.update({'validation_MNLL': validation_mean_NLL, 'validation_loss': validation_loss,
                                           'percent_right': percent_right_choices})
            self.record(epoch, **convergence_params)
        return validation_error

    def run_epoch(self, data, epoch):
        lr = self.lr.update(epoch)
        for row in data:
            user, item, rating = row
            prediction = self.predict_on_pair(user,item)
            error = rating-prediction
            u_b_gradient = (error - self.l2_users_bias * self.user_biases[user])
            i_b_gradient = (error - self.l2_items_bias * self.item_biases[item])
            self.user_biases[user] += lr * self.user_biases_gradient.get(u_b_gradient, user)
            self.item_biases[item] += lr * self.item_biases_gradient.get(i_b_gradient, item)
            self.global_bias += lr * error
            if epoch > self.number_bias_epochs:
                u_grad = (error * self.V[item, :] - self.l2_users * self.U[user, :])
                v_grad = (error * self.U[user, :] - self.l2_items * self.V[item, :])
                self.U[user, :] += lr * self.users_h_gradient.get(u_grad, user)
                self.V[item, :] += lr * self.items_h_gradient.get(v_grad, item)
            """
                        prediction = self.global_bias + self.user_biases[user] + self.item_biases[item] \
                       + self.U[user, :].dot(self.V[item, :].T)
            constant_term = np.exp(-prediction) / (1 + np.exp(-prediction))
            if rating == 0:
                constant_term -= 1
            constant_term *= -1
            u_b_gradient = (constant_term + self.l2_users_bias * self.user_biases[user])
            i_b_gradient = (constant_term + self.l2_items_bias * self.item_biases[item])
            self.user_biases[user] -= lr * self.user_biases_gradient.get(u_b_gradient, user)
            self.item_biases[item] -= lr * self.item_biases_gradient.get(i_b_gradient, item)
            self.global_bias -= lr * constant_term
            if epoch > self.number_bias_epochs:
                u_grad = (constant_term * self.V[item, :] + self.l2_users * self.U[user, :])
                v_grad = (constant_term * self.U[user, :] + self.l2_items * self.V[item, :])
                self.U[user, :] -= lr * self.users_h_gradient.get(u_grad, user)
                self.V[item, :] -= lr * self.items_h_gradient.get(v_grad, item)
            """


    def predict_which_item_more_likely(self, data):
        counter = 0
        for row in data.values:
            user, positive_item, negative_item = row
            counter += self.predict_on_pair(user, positive_item) >= self.predict_on_pair(user, negative_item)
        return counter / data.shape[0]
