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


class InferenceDataset(data.Dataset):
    def __init__(self, path = './', split='test'):
        self.path = path
        dataset_name = os.path.basename(path)
        self.json_path = os.path.join(path, dataset_name + '.json')
        self.pkl_path = os.path.join(path, dataset_name + '.pkl')
        if not os.path.exists(path):
            raise "no such file:{} !!!".format(path)
        else:
            data = json.load(open(self.json_path))

        self.data_dict = self._load_data_dict()

        self.data = []
        for temp in data:
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
                                'category': 'inference',
                                'chain_mask': np.ones(seq_length),
                                'chain_encoding': np.ones(seq_length),
                                'orig_surface': data['surface'],
                                'surface': normalize_coordinates(data['surface']),
                                'features': data['features'][:, :2],
                                })

    def _load_data_dict(self):
        with open(self.pkl_path, 'rb') as f:
            return pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]