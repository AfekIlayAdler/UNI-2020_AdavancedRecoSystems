import numpy as np


class MatrixFactorizationWithBiases:
    def __init__(self, config):
        self.n_users = config.n_users
        self.n_items = config.n_items
        self.h_len = config.hidden_dimension
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
        self.beta = 0.9

    def weight_init(self):
        self.U = np.random.normal(scale=1. / self.h_len, size=(self.n_users, self.h_len))
        self.V = np.random.normal(size=(self.n_items, self.h_len))

        self.users_h_gradient = np.zeros((self.n_users, self.h_len))
        self.items_h_gradient = np.zeros((self.n_items, self.h_len))
        # Initialize the biases
        self.user_biases = np.zeros(self.n_users)
        self.item_biases = np.zeros(self.n_items)

    def fit(self, train: np.array, validation: np.array):
        "data columns: [user id,movie_id,rating in 1-5]"
        self.weight_init()
        self.global_bias = np.mean(train[:, 2])
        for epoch in range(1, self.epochs + 1):
            np.random.shuffle(train)
            self.run_epoch(train, epoch)
            print(f"epoch number {epoch}")
            print(f"train error: {self.calc_mse(train)}")
            print(f"validation error: {self.calc_mse(validation)}")

    def calc_mse(self, x):
        e = 0
        for row in x:
            user, item, rating = row
            e += np.square(rating - self.predict_on_pair(user, item))
        return np.sqrt(e) / x.shape[0]

    def run_epoch(self, data, epoch):
        for row in data:
            user, item, rating = row
            prediction = self.predict_on_pair(user, item)
            # error = np.clip(rating - prediction, -1000, 1000)
            error = rating - prediction
            self.user_biases[user] += self.lr * (error - self.l2_users_bias * self.user_biases[user])
            self.item_biases[item] += self.lr * (error - self.l2_items_bias * self.item_biases[item])
            if epoch > self.number_bias_epochs:
                u_grad = (error * self.V[item, :] - self.l2_users * self.U[user, :])
                v_grad = (error * self.U[user, :] - self.l2_items * self.V[item, :])
                if self.momentum:
                    u_grad = self.beta * self.users_h_gradient[user, :] + (1 - self.beta) * u_grad
                    v_grad = self.beta * self.items_h_gradient[item, :] + (1 - self.beta) * v_grad
                    self.users_h_gradient[user, :] = u_grad
                    self.items_h_gradient[item, :] = v_grad
                self.U[user, :] += self.lr * u_grad
                self.V[item, :] += self.lr * v_grad

    def predict_on_pair(self, user, item):
        return self.global_bias + self.user_biases[user] + self.item_biases[item] \
               + self.U[user, :].dot(self.V[item, :].T)
