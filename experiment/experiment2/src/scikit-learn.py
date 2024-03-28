import numpy as np
import pandas as pd
from sklearn.neighbors import\
KNeighborsClassifier as KNN
from data import IrisData

args = dict()
args['dataroot'] = r'./data/'
args['predict'] = r'./prediction/'

class SciIrisData(IrisData):

    def __init__(self, dataroot):
        super().__init__(dataroot)

    def euclidean(self, k):
        model = KNN(k)
        data = self.train[:, :-1]
        label = self.train[:, -1].astype(np.int32)
        model.fit(data, label)
        data = self.val[:, :-1]
        label = self.val[:, -1].astype(np.int32)
        y_pred = model.predict(data)
        acc = np.sum(y_pred == label)
        acc /= data.shape[0]
        return model, acc

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')

    iris = SciIrisData(args['dataroot'])
    k = 5

    model, acc = iris.euclidean(k)
    print('euclidean accuracy: {}'.format(acc))
    y_pred = model.predict(iris.test)
    pd.DataFrame(y_pred).to_csv(
        args['predict'] + 'task3_test_prediction.csv', 
        index=False, header=['label']
    )
