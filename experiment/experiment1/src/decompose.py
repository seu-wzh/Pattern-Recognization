import numpy as np

class PCA(object):

    def __init__(self, n_components):
        self.n = n_components
    
    def fit(self, data):
        self.mean = np.mean(data, axis=0)
        x = data - self.mean
        scale = np.matmul(x.T, x)
        values, vecs = np.linalg.eig(scale)
        self.transformer = vecs[:, :self.n]
    
    def transform(self, data):
        return np.matmul(
            data - self.mean, 
            self.transformer
        )
    
class LDA(object):

    def __init__(self, n_components):
        self.n = n_components

    def fit(self, data, label, c, epsilon=0.01):
        data_split = [
            data[np.where(label == i)]
            for i in range(c)
        ]
        self.mean = np.mean(data, axis=0)
        mean_split = [
            np.mean(data_i, axis=0)
            for data_i in data_split
        ]
        Sw = list()
        Sb = list()
        for data_i, mean_i in zip(data_split, mean_split):
            Sw.append(np.matmul(
                    (data_i - mean_i).T, 
                    (data_i - mean_i)
            ))
            Sb.append(len(data_i) * np.outer(
                    mean_i - self.mean, 
                    mean_i - self.mean
            ))
        Sw = np.sum(Sw, axis=0)
        Sb = np.sum(Sb, axis=0)
        if np.linalg.matrix_rank(Sw) < Sw.shape[0]:
            Sw += epsilon * np.eye(Sw.shape[0])
        scale = np.matmul(np.linalg.inv(Sw), Sb)
        values, vecs = np.linalg.eig(scale)
        values = values.astype(np.float32)
        vecs   = vecs.astype(np.float32)
        self.transformer = vecs[:, :self.n]

    def transform(self, data, scale=1000):
        return scale * np.matmul(
            data - self.mean, 
            self.transformer
        )