import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import\
LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from data import WineData

args = dict()
args['dataroot'] = r'./data/'

class SciWineData(WineData):

    def __init__(self, dataroot:str):
        super().__init__(dataroot)

    def decompose(self, solver, n_components):
        model = solver(n_components=n_components)
        model.fit(self.data, self.label)
        self.data = model.transform(self.data)
        self.dim = n_components
        return model
    
    def classify(self):
        model = LogisticRegression()
        model.fit(self.data, self.label)
        y_pred = model.predict(self.data)
        acc = np.sum(y_pred == self.label)
        acc /= self.data.shape[0]
        return model, acc
    
if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')

    plt.style.use('seaborn')

    # *************** origin ******************** #

    wine_data = SciWineData(args['dataroot'])
    model, acc = wine_data.classify()
    print('origin data accuracy: {}'.format(acc))
    
    # ****************** PCA ******************** #

    wine_data = SciWineData(args['dataroot'])
    wine_data.decompose(PCA, 2)
    wine_data.render_data(
        'PCA', 
        (-150, 200), 
        (-50, 100)
    )
    model, acc = wine_data.classify()
    print('PCA data accuracy: {}'.format(acc))

    # ****************** LDA ******************** #

    wine_data = SciWineData(args['dataroot'])
    wine_data.decompose(LDA, 1)
    wine_data.render_data('LDA')
    model, acc = wine_data.classify()
    print('LDA data accuracy: {}'.format(acc))