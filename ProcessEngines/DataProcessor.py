import numpy as np

class DataLoader:
    def __init__(self, X: np.ndarray, y: np.ndarray, batch_size: int = 32, shuffle: bool = True, drop_last: bool = False):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_samples = X.shape[0]
        self.rng = np.random.default_rng()

        def __iter__(self):
            indices = np.arange(self.num_samples)
            
            num_batches = self.num_samples // self.batch_size if drop_last else int(np.ceil(self.num_samples / self.batch_size))
            
            if self.shuffle:
                self.rng.shuffle(indices)
            
            for i in range(num_batches):
                start_idx = i * self.batch_size
                end_idx = start_idx + self.batch_size
                batch_indices = indices[start_idx : end_idx]
                
                yield self.X[batch_indices], self.y[batch_indices]