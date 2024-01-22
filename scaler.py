class Scaler3D:

    def fit(self, X):
        self.mean = X.reshape(-1 ,X.shape[-1]).mean(0).reshape(1 ,1 ,-1)
        self.std = X.reshape(-1 ,X.shape[-1]).std(0).reshape(1 ,1 ,-1)
        return self

    def transform(self, X):
        return (X - self.mean) / self.std

    def inverse_transform(self, X):
        return (X * self.std) + self.mean


