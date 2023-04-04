class EarlyStopper:
    def __init__(self, patience=10, delta=1e-4):
        self.patience = patience
        self.delta = delta

        self._counter = 0

        self._best_info = None
        self._best_score = None

    def is_done(self):
        if self.patience >= 0:
            return self._counter >= self.patience
        return False

    def info(self):
        return self._best_info

    def __call__(self, score, info):
        assert not self.is_done()

        if self._best_score is None:
            self._best_score = score
            self._best_info = info
        elif score < self._best_score + self.delta:
            self._counter += 1
        else:
            self._best_score = score
            self._best_info = info
            self._counter = 0
