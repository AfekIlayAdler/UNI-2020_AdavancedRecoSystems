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


    def fit(self, train: np.array, validation: np.array, user_map: dict, item_map: dict):
        pass

    def predict(self, user, item):
        """
        predict on user and item with their original ids not internal ids
        """
        pass

    def predict_on_pair(self, user, item):
        pass

    def predict_on_new_user_existing_item(self, item):
        pass

    def predict_on_existing_user_new_item(self, user):
        pass

    def predict_on_new_user_new_item(self):
        pass
