import numpy as np
from numpy.linalg import solve
from HW1.matrix_factorization_abstract import MatrixFactorizationWithBiases
from HW1.momentum_wrapper import MomentumWrapper1D, MomentumWrapper2D

class MatrixFactorizationWithBiasesALS(MatrixFactorizationWithBiases):
    def __init__(self, config):
        super().__init__(config.seed, config.hidden_dimension)
        self.n_users = config.n_users
        self.n_items = config.n_items
        self.ratings = config.ratings
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
        self.results = {}
        # TODO understand if we need self.batch_size

    def weight_init(self, user_map, item_map):
        self.user_map, self.item_map = user_map, item_map
        # TODO understand if we can get a better initialization
        self.U = np.random.normal(scale=1. / self.h_len, size=(self.n_users, self.h_len))
        self.V = np.random.normal(scale=1. / self.h_len, size=(self.n_items, self.h_len))
        # Initialize the biases
        self.user_biases = np.zeros(self.n_users)
        self.item_biases = np.zeros(self.n_items)

    def als_step(self, type_vec='user'):
        """
        One of the two ALS steps. Solve for the latent vectors
        specified by type.
        """
        if type_vec == 'user':
            latent_vectors = self.U
            fixed_vecs = self.V
            # Precompute
            YTY = fixed_vecs.T.dot(fixed_vecs)
            lambdaI = np.eye(YTY.shape[0]) * self.l2_users

            for u in range(latent_vectors.shape[0]):
                user_ranking = self.ratings.getrow(u)
                current_user_bias_vector = np.full((1, self.n_items), self.user_biases[u])
                global_bias_vector = np.full((1, self.n_items), self.global_bias)
                A = (YTY + lambdaI)
                reg = global_bias_vector+current_user_bias_vector+self.item_biases
                user_ranking.data -= np.take(reg.flatten(), user_ranking.indices)
                B = user_ranking.dot(fixed_vecs).T
                latent_vectors[u, :] = solve(A, B).reshape(self.h_len)

        elif type_vec == 'item':
            latent_vectors = self.V
            fixed_vecs = self.U
            # Precompute
            XTX = fixed_vecs.T.dot(fixed_vecs)
            lambdaI = np.eye(XTX.shape[0]) * self.l2_items

            for i in range(latent_vectors.shape[0]):
                item_ranking = self.ratings.getcol(i)  # get the user ranking vector
                current_item_bias_vector = np.full((1, self.n_users), self.item_biases[i])
                global_bias_vector = np.full((1, self.n_users), self.global_bias)
                A = (XTX + lambdaI)
                reg = global_bias_vector+self.user_biases+current_item_bias_vector
                item_ranking.data -= np.take(reg.flatten(), item_ranking.indices)
                B = item_ranking.T.dot(fixed_vecs).T
                latent_vectors[i, :] = solve(A, B).reshape(self.h_len)
        return latent_vectors

    def update_bias(self, type_vec='user'):
        if type_vec == 'user':
            for u in range(self.n_users):
                user_ranking = self.ratings.getrow(u)
                global_bias_vector = np.full((1, self.n_items), self.global_bias)
                A = user_ranking.count_nonzero() + self.l2_users_bias
                reg = global_bias_vector + self.item_biases + np.dot(self.U[u], self.V.T)
                user_ranking.data -= np.take(reg.flatten(), user_ranking.indices)
                self.user_biases[u] = np.sum(user_ranking)/A
            return self.user_biases

        elif type_vec == 'item':
            for i in range(self.n_items):
                item_ranking = self.ratings.getcol(i)
                global_bias_vector = np.full((1, self.n_users), self.global_bias)
                A = item_ranking.count_nonzero() + self.l2_items_bias
                reg = global_bias_vector + self.user_biases + np.dot(self.U, self.V[i])
                item_ranking.data -= np.take(reg.flatten(), item_ranking.indices)
                self.item_biases[i] = np.sum(item_ranking)/A
            return self.item_biases


    def run_epoch(self, n_iter):
        """
        Train model for n_iter iterations. Can be
        called multiple times for further training.
        """
        ctr = 1
        while ctr <= n_iter:
            if ctr % 10 == 0:
                print ('\tcurrent iteration: {}'.format(ctr))

            self.U = self.als_step('user')
            self.V = self.als_step('item')
            self.user_biases = self.update_bias('user')
            self.item_biases = self.update_bias('item')

            ctr += 1

    def fit(self, train: np.array, validation: np.array, user_map: dict, item_map: dict):
        """data columns: [user id,movie_id,rating in 1-5]"""
        self.weight_init(user_map, item_map)
        self.global_bias = np.mean(train[:, 2])
        iter_array = [10]
        iter_diff = 0
        for (i, n_iter) in enumerate(iter_array):
            self.run_epoch(n_iter - iter_diff)
            self.record(n_iter, train_accuracy=self.prediction_loss(train),
                        test_accuracy=self.prediction_loss(validation),
                        train_loss=self.calc_loss(train), test_loss=self.calc_loss(validation))

