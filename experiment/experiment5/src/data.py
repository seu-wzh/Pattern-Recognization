import numpy as np

class DataLoader(object):

    def __init__(
            self, 
            dataroot:str, 
            files:dict, 
            batch_size:int, 
            train:bool=True
    ):
        self.batch_size = batch_size
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

    def __iter__(self):
        return self
    
    def __next__(self):
        index  = np.random.randint(
            self.size, 
            size=self.batch_size
        )
        images = self.data['images'][index]
        labels = self.data['labels'][index]
        labels = np.eye(10)[labels]
        return images / 255, labels