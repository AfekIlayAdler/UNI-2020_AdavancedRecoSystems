class LearningRateScheduler:
    def __init__(self, lr=0.01, decay=0.9):
        self.lr = lr
        self.decay = decay

    def update(self, n):
        if n > 1:
            self.lr *= self.decay
        return self.lr


class EarlyStopping:
    def __init__(self, n_iter, min_epoch=15):
        """
        :param n_iter: if error is increasing for n_iter -> stop
        :param min_epoch: don't stop before min_epcoch
        """
        self.n_iter = n_iter
        self.last_value = 0
        self.consecutive_increasing_errors = 0
        self.min_epoch = min_epoch

    def stop(self, epoch, error):
        if error > self.last_value and epoch >= self.min_epoch:
            self.consecutive_increasing_errors += 1
        if self.consecutive_increasing_errors >= self.n_iter:
            return True
        self.last_value = error
        return False


