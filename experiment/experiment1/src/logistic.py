import numpy as np

def sigmoid(x): 
    return 1 / (1 + np.exp(-x))

class LogisticRegression(object):

    def __init__(self, input_dim, scale=0.1) -> None:
        self.w = scale * np.random.randn(input_dim)
        self.b = scale * np.random.randn()
    
    def predict(self, data, logit=True):
        y_pred = np.matmul(data, self.w)
        y_pred = y_pred + self.b
        if logit:
            y_pred = sigmoid(y_pred)
        else:
            y_pred = y_pred > 0
        return y_pred
    
    def fit(
            self, 
            data, 
            label, 
            lr, 
            decay, 
            iterations, 
            patience, 
            epsilon=0.001
    ): 
        history = dict()
        history['loss'] = []
        history['acc']  = []
        n = data.shape[0]
        best_acc = 0.
        tolerance = 0
        for i in range(iterations):
            y_pred = self.predict(data)
            positive = label * (1 / (y_pred + epsilon))
            negative = (1 - label) * (1 / (y_pred - 1 - epsilon))
            grad = -(positive + negative)
            grad = grad * y_pred * (1 - y_pred)
            grad_w = np.matmul(data.T, grad) / n
            grad_b = np.sum(grad) / n
            self.w -= lr * grad_w
            self.b -= lr * grad_b
            positive = label * np.log(y_pred + epsilon)
            negative = (1 - label) * np.log(1 - y_pred + epsilon)
            loss = -np.sum(positive + negative) / n
            y_pred = y_pred > 0.5
            acc = np.sum(y_pred == label) / n
            if acc > best_acc:
                tolerance = 0
                best_acc = acc
            else:
                tolerance += 1
            if tolerance >= patience:
                break
            lr *= decay
            history['loss'].append(loss)
            history['acc'].append(acc)
        return history