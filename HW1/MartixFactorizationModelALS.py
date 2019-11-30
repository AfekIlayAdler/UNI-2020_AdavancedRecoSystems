import numpy as np

from HW1.matrix_factorization_abstract import MatrixFactorizationWithBiases
from HW1.momentum_wrapper import MomentumWrapper1D, MomentumWrapper2D

class MatrixFactorizationWithBiasesALS(MatrixFactorizationWithBiases):
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
        self.user_biases_gradient = None
        self.item_biases_gradient = None
        self.beta = 0.7
        self.results = {}
        # TODO understand if we need self.batch_size

    def weight_init(self):
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
        self.weight_init()