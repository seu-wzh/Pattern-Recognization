import numpy as np
import pandas as pd


class IrisData(object):

    def __init__(self, dataroot):
        self.train = dataroot + 'train.csv'
        self.val = dataroot + 'val.csv'
        self.test = dataroot + 'test_data.csv'
        self.train = pd.read_csv(self.train).to_numpy()
        self.val = pd.read_csv(self.val).to_numpy()
        self.test = pd.read_csv(self.test).to_numpy()