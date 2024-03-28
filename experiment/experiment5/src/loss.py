import numpy as np

class Loss:

    def __init__(self):
        pass
    def __call__(
            self, 
            y_pred, 
            y_true, 
            grad:bool=True
    ):
        pass

class SparseCrossEntropy(Loss):

    def __init__(self, epsilon:float=0.01):
        super().__init__()
        self.epsilon = epsilon
    
    def __call__(
            self, 
            y_pred, 
            y_true, 
            grad:bool=True
    ):
        loss = -y_true * np.log(y_pred + self.epsilon)
        loss = np.sum(loss)
        if grad:
            dy = -y_true / (y_pred + self.epsilon)
            return loss, dy
        return loss