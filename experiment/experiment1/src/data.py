import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class WineData(object):

    def __init__(
            self, 
            dataroot:str
    ):
        self.data  = dict()
        self.label = dict()

        filename = 'winequality-red.csv'
        filename = dataroot + filename
        self.data['red'] = pd.read_csv(filename, sep=';')
        filename = 'winequality-white.csv'
        filename = dataroot + filename
        self.data['white'] = pd.read_csv(filename, sep=';')

        self.label['red'] = np.zeros(
            self.data['red'].shape[0]
        )
        self.label['white'] = np.ones(
            self.data['white'].shape[0]
        )

        self.data = np.concatenate([
            self.data['red'], 
            self.data['white']
        ], axis=0)
        self.label = np.concatenate([
            self.label['red'], 
            self.label['white']
        ], axis=0)

        self.dim = self.data.shape[-1]
        

    def render_data(self, title, xlim=None, ylim=None):
        if self.dim != 2:
            return
        fig = plt.figure(figsize=(12, 12))
        axs = plt.subplot(1, 1, 1)
        axs.set_title(title, fontsize=20)
        if xlim is not None:
            axs.set_xlim(xlim[0], xlim[1])
        if ylim is not None:
            axs.set_ylim(ylim[0], ylim[1])
        red = self.data[self.label == 0]
        white = self.data[self.label == 1]
        axs.scatter(
            red[:, 0], 
            red[:, 1], 
            s=5, c='darkcyan', 
            label='red'
        )
        axs.scatter(
            white[:, 0], 
            white[:, 1], 
            s=5, c='orange', 
            label='white'
        )
        axs.legend(fontsize=20)
        plt.show()