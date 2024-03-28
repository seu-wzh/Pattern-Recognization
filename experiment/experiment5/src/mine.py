import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from data import DataLoader
from module import Linear
from module import LayerNormalization
from module import ReLU
from module import Sigmoid
from module import Tanh
from module import Softmax
from module import Sequence
from module import MLP
from loss import SparseCrossEntropy

args = dict()
args['dataroot'] =  r'./data/'
args['files'] = {
    'train': {
        'images': 'train-images.idx3-ubyte', 
        'labels': 'train-labels.idx1-ubyte'
    }, 
    'val': {
        'images': 't10k-images.idx3-ubyte', 
        'labels': 't10k-labels.idx1-ubyte'
    }
}
args['modelroot'] = r'./model/'
args['lr'] = 0.01
args['epochs'] = 32
args['batch_nbr'] = 940
args['batch_size'] = 64
args['decay'] = 0.97
args['patience'] = 5

def train_on_batch(
        lr, 
        model, 
        loader, 
        criterion
):
    images, labels = next(loader)
    images = images.reshape((-1, 28*28))
    y_pred = model.forward(images)
    loss, dy = criterion(y_pred, labels)
    model.zero_grad()
    model.backward(dy)
    model.update(lr)
    y_pred = np.argmax(y_pred, axis=-1)
    labels = np.argmax(labels, axis=-1)
    correct = np.sum(y_pred == labels)
    return loss, correct

def val_on_batch(
        model, 
        loader, 
        criterion
):
    images, labels = next(loader)
    images = images.reshape((-1, 28*28))
    y_pred = model.forward(images)
    loss = criterion(y_pred, labels, grad=False)
    y_pred = np.argmax(y_pred, axis=-1)
    labels = np.argmax(labels, axis=-1)
    correct = np.sum(y_pred == labels)
    return loss, correct

def train(
        model, 
        loader, 
        criterion, 
        lr, 
        epochs, 
        batch_nbr, 
        batch_size, 
        decay, 
        patience
):
    history = {
        'train': {
            'loss': [], 
            'acc': []
        }, 
        'val': {
            'loss': [], 
            'acc': []
        }
    }
    best_acc = 0.0
    tolerance = 0
    sample_nbr = batch_nbr * batch_size
    for epoch in tqdm(range(epochs)):
        history['train']['loss'].append(0.)
        history['train']['acc'].append(0.)
        history['val']['loss'].append(0.)
        history['val']['acc'].append(0.)
        for batch in range(batch_nbr):
            loss, correct = train_on_batch(
                lr, 
                model, 
                loader['train'], 
                criterion
            )
            history['train']['loss'][-1] += loss
            history['train']['acc'][-1] += correct
        for batch in range(batch_nbr):
            loss, correct = val_on_batch(
                model, 
                loader['val'], 
                criterion
            )
            history['val']['loss'][-1] += loss
            history['val']['acc'][-1] += correct
        lr *= decay
        history['train']['loss'][-1] /= sample_nbr
        history['train']['acc'][-1]  /= sample_nbr
        history['val']['loss'][-1]   /= sample_nbr
        history['val']['acc'][-1]    /= sample_nbr
        if history['val']['acc'][-1] > best_acc:
            best_acc = history['val']['acc'][-1]
            tolerance = 0
        else:
            tolerance += 1
        if tolerance > patience:
            break
    return history

def render_history(history):
    fig = plt.figure(figsize=(13, 6))
    axs = (
        plt.subplot(1, 2, 1), 
        plt.subplot(1, 2, 2)
    )
    axs[0].set_title('Loss', fontsize=20)
    axs[1].set_title('Accuracy', fontsize=20)
    axs[0].set_xlabel('iteration', fontsize=20)
    axs[1].set_xlabel('iteration', fontsize=20)
    axs[0].plot(
        history['train']['loss'], 
        color='darkcyan', 
        label='train'
    )
    axs[0].plot(
        history['val']['loss'], 
        color='orange', 
        label='val'
    )
    axs[1].plot(
        history['train']['acc'], 
        color='darkcyan', 
        label='train'
    )
    axs[1].plot(
        history['val']['acc'], 
        color='orange', 
        label='val'
    )
    axs[0].legend(fontsize=20)
    axs[1].legend(fontsize=20)
    plt.show()

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')

    plt.style.use('seaborn')

    loader = dict()
    loader['train'] = DataLoader(
        args['dataroot'], 
        args['files'], 
        args['batch_size'], 
        train=True
    )
    loader['val'] = DataLoader(
        args['dataroot'], 
        args['files'], 
        args['batch_size'], 
        train=False
    )
    model = Sequence([
        Linear(28*28, 10), 
        Softmax(axis=-1)
    ])
    criterion = SparseCrossEntropy()
    history = train(
        model, 
        loader, 
        criterion, 
        args['lr'], 
        args['epochs'], 
        args['batch_nbr'], 
        args['batch_size'], 
        args['decay'], 
        args['patience']
    )
    render_history(history)
    model.save_weights(args['modelroot'])