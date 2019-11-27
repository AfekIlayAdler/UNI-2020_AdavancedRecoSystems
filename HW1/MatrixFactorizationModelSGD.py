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
        self.momentum = False
        self.users_h_gradient = None  # for momentum
        self.items_h_gradient = None  # for momentum
        self.user_map = None
        self.item_map = None
        self.beta = 0.9
        # TODO understand if we need self.batch_size

    def weight_init(self, user_map, item_map):
        self.user_map, self.item_map = user_map, item_map
        # TODO understand if we can get a better initialization
        self.U = np.random.normal(scale=1. / self.h_len, size=(self.n_users, self.h_len))
        self.V = np.random.normal(scale=1. / self.h_len, size=(self.n_items, self.h_len))

        self.users_h_gradient = np.zeros((self.n_users, self.h_len))
        self.items_h_gradient = np.zeros((self.n_items, self.h_len))
        # Initialize the biases
        self.user_biases = np.zeros(self.n_users)
        self.item_biases = np.zeros(self.n_items)

    def fit(self, train: np.array, validation: np.array, user_map: dict, item_map: dict):
        "data columns: [user id,movie_id,rating in 1-5]"
        self.weight_init(user_map, item_map)
        self.global_bias = np.mean(train[:, 2])
        for epoch in range(1, self.epochs + 1):
            np.random.shuffle(train)
            self.run_epoch(train, epoch)
            print(f"epoch number {epoch}")
            print(f"train error: {self.calc_rmse(train)}")
            print(f"validation error: {self.calc_rmse(validation)}")
            if epoch == 3:
                self.lr = self.lr / 10

    def calc_rmse(self, x):
        e = 0
        for row in x:
            user, item, rating = row
            e += np.square(rating - self.predict_on_pair(user, item))
        return np.sqrt(e) / x.shape[0]

    def run_epoch(self, data, epoch):
        for row in data:
            user, item, rating = row
            prediction = self.predict_on_pair(user, item)
            error = rating - prediction
            self.user_biases[user] += self.lr * (error - self.l2_users_bias * self.user_biases[user])
            self.item_biases[item] += self.lr * (error - self.l2_items_bias * self.item_biases[item])
            if epoch > self.number_bias_epochs:
                u_grad = (error * self.V[item, :] - self.l2_users * self.U[user, :])
                v_grad = (error * self.U[user, :] - self.l2_items * self.V[item, :])
                if self.momentum:
                    # TODO make momentum better in the first time
                    u_grad = self.beta * self.users_h_gradient[user, :] + (1 - self.beta) * u_grad
                    v_grad = self.beta * self.items_h_gradient[item, :] + (1 - self.beta) * v_grad
                    self.users_h_gradient[user, :] = u_grad
                    self.items_h_gradient[item, :] = v_grad
                self.U[user, :] += self.lr * u_grad
                self.V[item, :] += self.lr * v_grad

    def predict(self, user, item):
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

    def predict_on_pair(self, user, item):
        # TODO make sure that if we see a new user we return the global mean and if we have a new item and an
        #  existing user exc exc
        return self.global_bias + self.user_biases[user] + self.item_biases[item] \
               + self.U[user, :].dot(self.V[item, :].T)

    def predict_on_new_user_existing_item(self, item):
        return self.global_bias + self.item_biases[item]

    def predict_on_existing_user_new_item(self, user):
        return self.global_bias + self.user_biases[user]

    def predict_on_new_user_new_item(self):
        return self.global_bias
