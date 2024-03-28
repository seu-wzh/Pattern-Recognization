import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from logistic import LogisticRegression
from decompose import PCA
from decompose import LDA
from data import WineData

args = dict()
args['dataroot'] = r'./data/'


class MyWineData(WineData):

    def __init__(self, dataroot:str):
        super().__init__(dataroot)

    def decompose(
            self, 
            solver, 
            n_components
    ):
        model = solver(n_components)
        if solver is PCA:
            model.fit(self.data)
        elif solver is LDA:
            model.fit(
                self.data, 
                self.label, 
                2
            )
        self.data = model.transform(self.data)
        self.dim = n_components
        return model

    def classify(
            self, 
            lr, 
            decay, 
            iterations, 
            patience
    ):
        model = LogisticRegression(self.dim)
        history = model.fit(
            self.data, 
            self.label, 
            lr, 
            decay, 
            iterations, 
            patience
        )
        return model, history
    
    def render_history(self, history):
        fig = plt.figure(figsize=(13, 6))
        axs = [
            plt.subplot(1, 2, 1), 
            plt.subplot(1, 2, 2)
        ]
        axs[0].set_title('loss', fontsize=20)
        axs[1].set_title('accuracy', fontsize=20)
        axs[0].plot(history['loss'], color='darkcyan')
        axs[1].plot(history['acc'],  color='orange')
        plt.show()

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')

    plt.style.use('seaborn')
    
    # *************** origin ******************** #

    wine_data = MyWineData(args['dataroot'])
    model, history = wine_data.classify(
        lr=0.01, 
        decay=0.97,
        iterations=32, 
        patience=5
    )
    wine_data.render_history(history)
    
    # ****************** PCA ******************** #

    wine_data = MyWineData(args['dataroot'])
    wine_data.decompose(PCA, 2)
    wine_data.render_data(
        'PCA', 
        (-150, 200), 
        (-100, 50)
    )
    model, history = wine_data.classify(
        lr=0.001, 
        decay=0.97,
        iterations=64, 
        patience=5
    )
    wine_data.render_history(history)

    # ****************** LDA ******************** #

    wine_data = MyWineData(args['dataroot'])
    wine_data.decompose(LDA, 2)
    wine_data.render_data(
        'LDA', 
        (-30, 20), 
        (-10, 5)
    )
    model, history = wine_data.classify(
        lr=0.01, 
        decay=0.97,
        iterations=64, 
        patience=5
    )
    wine_data.render_history(history)