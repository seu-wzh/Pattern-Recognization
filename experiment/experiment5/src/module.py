import numpy as np

class Module:

    def __init__(self):
        pass
    def forward(self, x):
        pass
    def backward(self, dy):
        pass
    def update(self, lr):
        pass
    def zero_grad(self):
        pass
    def save_weights(self, modelroot:str):
        pass
    def load_weights(self, modelroot:str):
        pass

class Linear(Module):

    def __init__(
            self, 
            input_dim:int, 
            output_dim:int, 
            init_scale=0.1
    ):
        super().__init__()
        self.w = np.random.randn(
            input_dim, 
            output_dim
        ) * init_scale
        self.b = np.random.randn(
            output_dim
        ) * init_scale
        self.dw = np.zeros_like(self.w)
        self.db = np.zeros_like(self.b)
        self.input = None

    def forward(self, x):
        self.input = np.copy(x)
        s = np.matmul(x, self.w)
        s = s + self.b
        return s

    def backward(self, dy):
        self.db += np.mean(dy, axis=0)
        self.dw += np.matmul(self.input.T, dy) / self.input.shape[0]
        return np.matmul(dy, self.w.T)
    
    def update(self, lr):
        self.w -= lr * self.dw
        self.b -= lr * self.db

    def zero_grad(self):
        self.dw = np.zeros_like(self.w)
        self.db = np.zeros_like(self.b)

    def save_weights(self, modelroot: str):
        np.save(modelroot + 'w.npy', self.w)
        np.save(modelroot + 'b.npy', self.b)

    def load_weights(self, modelroot: str):
        self.w = np.load(modelroot + 'w.npy')
        self.b = np.load(modelroot + 'b.npy')

class LayerNormalization(Module):

    def __init__(self, axis, epsilon=0.001):
        super().__init__()
        self.mean = 0.
        self.var  = 0.
        self.axis = axis
        self.epsilon = epsilon

    def forward(self, x):
        self.mean = np.mean(
            x, 
            axis=self.axis, 
            keepdims=True
        )
        self.var = np.var(
            x, 
            axis=self.axis, 
            keepdims=True
        ) + self.epsilon
        s = (x - self.mean) / self.var
        return s
    
    def backward(self, dy):
        return dy / self.var
        

class ReLU(Module):

    def __init__(self):
        super().__init__()
        self.input = None

    def forward(self, x):
        self.input = np.copy(x)
        return np.maximum(0, x)
    
    def backward(self, dy):
        dy[self.input <= 0] = 0
        return dy

class Sigmoid(Module):

    def __init__(self):
        super().__init__()
        self.output = None
    
    def forward(self, x):
        self.output = 1 / (1 + np.exp(-x))
        return self.output
    
    def backward(self, dy):
        return dy * self.output * (1 - self.output)
    
class Tanh(Module):

    def __init__(self):
        super().__init__()
        self.output = None
    
    def forward(self, x):
        self.output = np.tanh(x)
        return self.output
    
    def backward(self, dy):
        return dy * (1 - self.output ** 2)

class Softmax(Module):

    def __init__(self, axis):
        super().__init__()
        self.axis = axis
        self.output = None

    def forward(self, x):
        s1 = np.max(x, axis=self.axis, keepdims=True)
        s1 = np.exp(x - s1)
        s2 = np.sum(s1, self.axis, keepdims=True)
        self.output = s1 / s2
        return self.output
    
    def backward(self, dy):
        return dy * self.output * (1 - self.output)
    
class Sequence(Module):

    def __init__(self, module_list):
        super().__init__()
        self.module_list = module_list
    
    def forward(self, x):
        for module in self.module_list:
            x = module.forward(x)
        return x
    
    def backward(self, dy):
        for module in reversed(self.module_list):
            dy = module.backward(dy)
        return dy
    
    def update(self, lr):
        for module in self.module_list:
            module.update(lr)

    def zero_grad(self):
        for module in self.module_list:
            module.zero_grad()
    
    def save_weights(self, modelroot: str):
        for i, module in enumerate(self.module_list):
            module.save_weights(
                modelroot + 'sequence{}_'.format(i)
            )
    
    def load_weights(self, modelroot: str):
        for i, module in enumerate(self.module_list):
            module.load_weights(
                modelroot + 'sequence{}_'.format(i)
            )

class MLP(Module):

    def __init__(
            self, 
            input_dim, 
            hidden_dim, 
            output_dim
    ):
        super().__init__()
        self.w1 = np.random.randn(
            input_dim, 
            hidden_dim
        )
        self.b1 = np.random.randn(
            hidden_dim
        )
        self.w2 = np.random.randn(
            hidden_dim, 
            output_dim
        )
        self.b2 = np.random.randn(
            output_dim
        )
        self.dw1 = np.zeros_like(self.w1)
        self.db1 = np.zeros_like(self.b1)
        self.dw2 = np.zeros_like(self.w2)
        self.db2 = np.zeros_like(self.b2)

    def forward(self, x):
        self.input = x
        x = np.matmul(x, self.w1) + self.b1
        x = np.tanh(x)
        self.hidden = x
        x = np.matmul(x, self.w2) + self.b2
        s1 = np.max(x, axis=-1, keepdims=True)
        s1 = np.exp(x - s1)
        s2 = np.sum(s1, axis=-1, keepdims=True)
        x = s1 / s2
        self.output = s1 / s2
        return x
    
    def backward(self, dy):
        dy = dy * self.output * (1 - self.output)
        self.db2 += np.mean(dy, axis=0)
        self.dw2 += np.matmul(self.hidden.T, dy) / self.hidden.shape[0]
        dy = np.matmul(dy, self.w2.T)
        dy = dy * (1 - self.hidden ** 2)
        self.db1 += np.mean(dy, axis=0)
        self.dw1 += np.matmul(self.input.T, dy) / self.hidden.shape[0]
        dy = np.matmul(dy, self.w1.T)
        return dy
    
    def update(self, lr):
        self.w1 -= lr * self.dw1
        self.b1 -= lr * self.db1
        self.w2 -= lr * self.dw2
        self.b2 -= lr * self.db2

    def zero_grad(self):
        self.dw1 = np.zeros_like(self.w1)
        self.db1 = np.zeros_like(self.b1)
        self.dw2 = np.zeros_like(self.w2)
        self.db2 = np.zeros_like(self.b2)