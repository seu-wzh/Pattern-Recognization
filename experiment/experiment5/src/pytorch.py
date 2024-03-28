import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from torch import tensor

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torch.nn import Module
from torch.nn import Linear
from torch.nn import Flatten
from torch.nn import ReLU
from torch.nn import Softmax
from torch.nn import Sequential
from torch.nn import CrossEntropyLoss

from torch.optim import SGD
from torch.optim.lr_scheduler import ExponentialLR

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
args['lr'] = 0.01
args['epochs'] = 32
args['batch_nbr'] = 940
args['batch_size'] = 64
args['decay'] = 0.95
args['patience'] = 5

class TorchData(Dataset):

    def __init__(
            self, 
            dataroot:str, 
            files:dict, 
            train:bool=True
    ):
        super().__init__()
        self.data = dict()
        data_type = 'train' if train else 'val'
        name = files[data_type]['images']
        with open(dataroot + name, 'rb') as f:
            self.data['images'] = np.frombuffer(
                f.read(), 
                dtype=np.uint8, 
                offset=16
            ).reshape(-1, 28, 28)
        name = files[data_type]['labels']
        with open(dataroot + name, 'rb') as f:
            self.data['labels'] = np.frombuffer(
                f.read(), 
                dtype=np.uint8, 
                offset=8
            )
        self.size = self.data['images'].shape[0]

    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        x = self.data['images'][index]
        y = self.data['labels'][index]
        x = tensor(x / 255, dtype=torch.float32)
        y = tensor(y, dtype=torch.long)
        return x, y
    
def train_on_epoch(
        model, 
        loader, 
        criterion, 
        optimizer
):
    history = dict()
    history['loss'] = 0.0
    history['acc']  = 0.0
    total = 0
    for sample in loader['train']:
        images, labels = sample
        optimizer.zero_grad()
        predict = model(images)
        loss = criterion(predict, labels)
        loss.backward()
        optimizer.step()
        total += images.size(0)
        history['loss'] += loss.item()
        predict = predict.argmax(dim=1)
        history['acc'] += (predict == labels).sum().item()
    history['loss'] /= total
    history['acc']  /= total
    return history

def val_on_epoch(
        model, 
        loader, 
        criterion
):
    history = dict()
    history['loss'] = 0.0
    history['acc']  = 0.0
    total = 0
    for sample in loader['val']:
        images, labels = sample
        with torch.no_grad():
            predict = model(images)
        loss = criterion(predict, labels)
        total += images.size(0)
        history['loss'] += loss.item()
        predict = predict.argmax(dim=1)
        history['acc'] += (predict == labels).sum().item()
    history['loss'] /= total
    history['acc']  /= total
    return history

def train(
        model, 
        loader, 
        criterion, 
        lr, 
        epochs, 
        decay, 
        patience
):
    history = dict()
    history['train'] = {'loss':[], 'acc':[]}
    history['val']   = {'loss':[], 'acc':[]}
    max_acc = 0.0
    tolerance = 0

    optimizer = SGD(
        model.parameters(), 
        lr=lr, 
        momentum=0.9
    )
    scheduler = ExponentialLR(
        optimizer, 
        gamma=decay
    )

    for epoch in tqdm(range(epochs)):
        epoch_history = train_on_epoch(
            model, 
            loader, 
            criterion, 
            optimizer
        )
        history['train']['loss'].append(
            epoch_history['loss']
        )
        history['train']['acc'].append(
            epoch_history['acc']
        )

        epoch_history = val_on_epoch(
            model, 
            loader, 
            criterion
        )
        history['val']['loss'].append(
            epoch_history['loss']
        )
        history['val']['acc'].append(
            epoch_history['acc']
        )

        if epoch_history['acc'] > max_acc:
            max_acc = epoch_history['acc']
            tolerance = 0
        else:
            tolerance += 1
        if tolerance > patience:
            break

        scheduler.step()
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

    dataset = dict()
    dataset['train'] = TorchData(
        args['dataroot'], 
        args['files'], 
        train=True
    )
    dataset['val'] = TorchData(
        args['dataroot'], 
        args['files'], 
        train=False
    )
    loader = dict()
    loader['train'] = DataLoader(
        dataset['train'], 
        args['batch_size'], 
        shuffle=True
    )
    loader['val'] = DataLoader(
        dataset['val'], 
        args['batch_size'], 
        shuffle=True
    )
    model = Sequential(
        Flatten(), 
        Linear(28*28, 10), 
        Softmax(dim=-1)
    )
    criterion = CrossEntropyLoss()
    history = train(
        model, 
        loader, 
        criterion, 
        args['lr'], 
        args['epochs'], 
        args['decay'], 
        args['patience']
    )
    render_history(history)