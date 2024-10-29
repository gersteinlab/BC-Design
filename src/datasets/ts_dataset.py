import os
import json
import numpy as np
import torch.utils.data as data
import pickle


def normalize_coordinates(surface):
    """
    Normalize the coordinates of the surface.
    """
    surface = np.array(surface)
    center = np.mean(surface, axis=0)
    max_ = np.max(surface, axis=0)
    min_ = np.min(surface, axis=0)
    length = np.max(max_ - min_)
    normalized_surface = (surface - center) / length
    return normalized_surface


class TS50Dataset(data.Dataset):
    def __init__(self, path = './', split='test'):
        self.path = path
        if not os.path.exists(path):
            raise "no such file:{} !!!".format(path)
        else:
            ts50_data = json.load(open(path+'/ts50.json'))

        self.data_dict = self._load_data_dict()

        # TS500 has proteins with lengths of 500+
        # TS50 only contains proteins with lengths less than 500
        self.data = []
        for temp in ts50_data:
            title = temp['name']
            data = self.data_dict[title]
            seq_length = len(temp['seq'])
            coords = np.array(temp['coords'])
            self.data.append({'title':title,
                                'seq':temp['seq'],
                                'CA':coords[:,1,:],
                                'C':coords[:,2,:],
                                'O':coords[:,3,:],
                                'N':coords[:,0,:],
                                'category': 'ts50',
                                'chain_mask': np.ones(seq_length),
                                'chain_encoding': np.ones(seq_length),
                                'orig_surface': data['surface'],
                                'surface': normalize_coordinates(data['surface']),
                                'features': data['features'][:, :2],
                                })

    def _load_data_dict(self):
        with open(self.path + f'/ts50.pkl', 'rb') as f:
            return pickle.load(f)

    def __len__(self):
        return len(self.data)
    
    def get_item(self, index):
        return self.data[index]

    def __getitem__(self, index):
        return self.data[index]


class TS500Dataset(data.Dataset):
    def __init__(self, path = './', split='test'):
        self.path = path
        if not os.path.exists(path):
            raise "no such file:{} !!!".format(path)
        else:
            ts500_data = json.load(open(path+'/ts500.json'))

        self.data_dict = self._load_data_dict()

        # TS500 has proteins with lengths of 500+
        # TS50 only contains proteins with lengths less than 500
        self.data = []
        for temp in ts500_data:
            title = temp['name']
            data = self.data_dict[title]
            seq_length = len(temp['seq'])
            coords = np.array(temp['coords'])
            self.data.append({'title':title,
                                'seq':temp['seq'],
                                'CA':coords[:,1,:],
                                'C':coords[:,2,:],
                                'O':coords[:,3,:],
                                'N':coords[:,0,:],
                                'category': 'ts500',
                                'chain_mask': np.ones(seq_length),
                                'chain_encoding': np.ones(seq_length),
                                'orig_surface': data['surface'],
                                'surface': normalize_coordinates(data['surface']),
                                'features': data['features'][:, :2],
                                })

    def _load_data_dict(self):
        with open(self.path + f'/ts500.pkl', 'rb') as f:
            return pickle.load(f)

    def __len__(self):
        return len(self.data)
    
    def get_item(self, index):
        return self.data[index]

    def __getitem__(self, index):
        return self.data[index]