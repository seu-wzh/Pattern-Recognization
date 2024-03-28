import numpy as np
from tqdm import tqdm

class KNN(object):

    def __init__(self, solver:str):
        self.solver = solver
    
    def fit(
            self, 
            data, 
            label, 
            **kwargs
    ):
        self.data  = data
        self.label = label
        self.c = np.max(label) + 1
        if self.solver == 'euclidean':
            return
        index_split = [
            np.where(label == i)[0]
            for i in range(self.c)
        ]
        n = data.shape[0]
        input_dim = data.shape[-1]
        output_dim = kwargs['transform_dim']
        self.A = 0.75 * np.random.randn(
            input_dim, 
            output_dim
        )
        iterations = kwargs['iterations']
        lr = kwargs['lr']
        decay = kwargs['decay']
        history = []
        for iteration in tqdm(range(iterations)):
            tmp = np.matmul(data, self.A)
            tmp = tmp[:, np.newaxis, :] - \
                  tmp[np.newaxis, :, :]
            d = np.sum(tmp ** 2, axis=-1)
            z = np.sum(np.exp(-d), axis=-1, keepdims=True)
            z = z - 1
            p = np.exp(-d) / (z + 0.001)
            grad = np.zeros_like(self.A)
            similarity = 0.
            for i in range(n):
                index_i = index_split[label[i]]
                tmp = np.zeros((input_dim, input_dim))
                for k in range(n):
                    if k == i: continue
                    tmp += p[i][k] * np.outer(
                        data[i] - data[k], 
                        data[i] - data[k]
                    )
                for j in index_i:
                    similarity += p[i][j]
                    grad_j = np.outer(
                        data[i] - data[j], 
                        data[i] - data[j]
                    )
                    grad_j = 2 * p[i][j] * (tmp - grad_j)
                    grad_j = np.matmul(grad_j, self.A)
                    grad += grad_j
            self.A += lr * grad / n
            lr *= decay
            history.append(similarity)
        return history

    def predict(self, data, k):
        if self.solver == 'euclidean':
            distance = self.data - data
        else:
            distance = np.matmul(
                self.data - data, self.A
            )
        distance = np.sum(
            distance ** 2, axis=-1
        )
        index = np.argpartition(
            distance, kth=k
        )
        neighbours = self.label[index[:k]]
        counts = dict()
        for label in neighbours:
            if label not in counts:
                counts[label] = 1
            else:
                counts[label] += 1
        cate, times = max(
            counts.items(), 
            key=lambda x: x[1]
        )
        return cate
    
    def transform(self, data):
        if self.solver == 'mahalanobis':
            return np.matmul(data, self.A)
        else:
            return data