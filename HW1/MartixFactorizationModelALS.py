import numpy as np
from HW1.matrix_factorization_abstract import MatrixFactorizationWithBiases

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
            for u in range(self.U.shape[0]):
                user_ranking = self.ratings.getrow(u)
                YTY = self.V[user_ranking.indices].T.dot(self.V[user_ranking.indices])  # choosing only the vectors from V  the the crurent user rated
                lambdaI = np.eye(YTY.shape[0]) * self.l2_users
                A = np.linalg.inv(YTY + lambdaI)
                biases = self.global_bias+self.user_biases[u]+self.item_biases[user_ranking.indices]  # summing all parts of biases
                user_ranking.data -= biases
                B = np.sum(np.multiply(user_ranking.data, self.V[user_ranking.indices].T), axis=1)
                self.U[u, :] = B.dot(A)

        elif type_vec == 'item':
            for i in range(self.V.shape[0]):
                item_ranking = self.ratings.getcol(i)  # get the user ranking vector
                XTX = self.U[item_ranking.indices].T.dot(self.U[item_ranking.indices])
                lambdaI = np.eye(XTX.shape[0]) * self.l2_items
                A = np.linalg.inv(XTX + lambdaI)
                biases = self.global_bias+self.user_biases[item_ranking.indices]+self.item_biases[i] # summing all parts of biases
                item_ranking.data -= biases
                B = np.sum(np.multiply(item_ranking.data.reshape(-1, 1), self.U[item_ranking.indices]), axis=0)
                self.V[i, :] = B.dot(A)

    def update_bias(self, type_vec='user'):
        if type_vec == 'user':
            for u in range(self.n_users):
                user_ranking = self.ratings.getrow(u)
                A = user_ranking.nnz + self.l2_users_bias
                biases = self.global_bias + self.item_biases[user_ranking.indices] + np.dot(self.U[u], self.V[user_ranking.indices].T)
                user_ranking.data -= biases
                self.user_biases[u] = np.sum(user_ranking)/A

        elif type_vec == 'item':
            for i in range(self.n_items):
                item_ranking = self.ratings.getcol(i)
                A = item_ranking.nnz + self.l2_items_bias
                biases = self.global_bias + self.user_biases[item_ranking.indices] + np.dot(self.U[item_ranking.indices], self.V[i])
                item_ranking.data -= biases
                self.item_biases[i] = np.sum(item_ranking)/A


    def run_epoch(self, epoch):
        """
        Train model. Can be called multiple times for further training.
        """
        self.als_step('user')
        self.als_step('item')
        self.update_bias('user')
        self.update_bias('item')

    def fit(self, train: np.array, validation: np.array, user_map: dict, item_map: dict):
        """data columns: [user id,movie_id,rating in 1-5]"""
        self.weight_init(user_map, item_map)
        self.global_bias = np.mean(train[:, 2])
        for epoch in range(1, self.epochs + 1):
            self.run_epoch(epoch)
            self.record(epoch, train_accuracy=self.prediction_loss(train),
                        test_accuracy=self.prediction_loss(validation),
                        train_loss=self.calc_loss(train), test_loss=self.calc_loss(validation))

