import os
import json
import pickle
import numpy as np
from tqdm import tqdm
import random
import torch.utils.data as data
from .utils import cached_property
from transformers import AutoTokenizer
from sklearn.neighbors import NearestNeighbors
import gc

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


def sample_if_needed(data_dict, max_length=2000):
    for key, value in data_dict.items():
        surface = value['surface']
        features = value['features']
        
        if len(surface) > max_length:
            indices = np.random.choice(len(surface), max_length, replace=False)
            value['surface'] = surface[indices]
            value['features'] = features[indices]
    
    return data_dict



class CATHAFDBDataset(data.Dataset):
    def __init__(self, path_cath='./', path_afdb='./', split='train', max_length=1000, test_name='All', data=None, removeTS=0, version=4.2):
        self.version = version
        self.path_cath = path_cath
        self.path_afdb = path_afdb
        self.mode = split
        self.max_length = max_length
        self.test_name = test_name
        self.removeTS = removeTS
        
        if self.removeTS:
            self.remove = json.load(open(self.path_cath + '/remove.json', 'r'))['remove']
        
        if data is None:
            self.metadata = self._load_metadata()
        else:
            self.metadata = data
        
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D", cache_dir="gaozhangyang/model_zoom/transformers")
        
        # Load the entire dictionary corresponding to the current mode
        self.data_dict = self._load_data_dict()
    def _load_metadata(self):
        alphabet = 'ACDEFGHIKLMNPQRSTVWY'
        alphabet_set = set([a for a in alphabet])
        metadata = []
        
        # Load the split JSON files
        with open(self.path_cath + '/chain_set_splits.json') as f:
            dataset_splits = json.load(f)
        
        # Handle specific test splits if needed
        if self.test_name == 'L100': 
            with open(self.path_cath + '/test_split_L100.json') as f:
                test_splits = json.load(f)
            dataset_splits['test'] = test_splits['test']

        if self.test_name == 'sc': 
            with open(self.path_cath + '/test_split_sc.json') as f:
                test_splits = json.load(f)
            dataset_splits['test'] = test_splits['test']

        # Select the appropriate split
        if self.mode == 'valid':
            valid_titles = set(dataset_splits['validation'])
        else:
            valid_titles = set(dataset_splits[self.mode])

        if not os.path.exists(self.path_cath):
            raise FileNotFoundError("No such file: {} !!!".format(self.path_cath))
        else:
            with open(self.path_cath + '/chain_set.jsonl') as f:
                lines = f.readlines()
            for line in tqdm(lines):
                entry = json.loads(line)
                if self.removeTS and entry['name'] in self.remove:
                    continue
                
                bad_chars = set([s for s in entry['seq']]).difference(alphabet_set)
                if len(bad_chars) == 0 and len(entry['seq']) <= self.max_length and entry['name'] in valid_titles:
                    entry['coords']['CA'] = np.array(entry['coords']['CA'])
                    entry['coords']['C'] = np.array(entry['coords']['C'])
                    entry['coords']['O'] = np.array(entry['coords']['O'])
                    entry['coords']['N'] = np.array(entry['coords']['N'])
                    # create a mask representing whether the position of any value of entry['coords']['CA'] or entry['coords']['C'] or entry['coords']['O'] or entry['coords']['N'] is nan or infinite
                    # sum them up and check if the values are inf or nan
                    coords = np.stack([
                        entry['coords']['CA'],
                        entry['coords']['C'],
                        entry['coords']['O'],
                        entry['coords']['N']
                    ], axis=1)  # shape: (L, 4, 3)
                    mask = np.isnan(coords).sum(axis=(1,2)) > 0
                    mask = mask | (np.isinf(coords).sum(axis=(1,2)) > 0)
                    # print('shape of mask', mask.shape)
                    # print('shape of entry[coords][CA]', len(entry['coords']['CA']))
                    # remove the positions where the mask is True
                    entry['coords']['CA'] = entry['coords']['CA'][~mask]
                    entry['coords']['C'] = entry['coords']['C'][~mask]
                    entry['coords']['O'] = entry['coords']['O'][~mask]
                    entry['coords']['N'] = entry['coords']['N'][~mask]
                    idx = np.where(~mask)[0]
                    entry['seq'] = ''.join([entry['seq'][i] for i in idx])
                    metadata.append({
                        'title': entry['name'],
                        'seq_length': len(entry['seq']),
                        'seq': entry['seq'],
                        'coords': entry['coords'],
                    })

        if self.mode == 'train':
            if not os.path.exists(self.path_afdb):
                raise "no such file:{} !!!".format(self.path_afdb)
            else:
                afdb_data = json.load(open(self.path_afdb+'/afdb-large4000.json'))

                for temp in tqdm(afdb_data):
                    title = temp['name']
                    seq_length = len(temp['seq'])
                    coords = np.array(temp['coords'])
                    coords_dict = {
                        'CA': coords[:,1,:],
                        'C': coords[:,2,:],
                        'O': coords[:,3,:],
                        'N': coords[:,0,:],
                    }
                    metadata.append({'title':title,
                                        'seq':temp['seq'],
                                        'seq_length': seq_length,
                                        'coords': coords_dict,
                                        })
        return metadata

    def _load_data_dict(self):
        # Load the appropriate pickle file based on the mode and keep it in memory
        def _downcast_float32_inplace(d):
            # Convert numeric arrays to float32 to reduce memory
            for _title, item in d.items():
                if not isinstance(item, dict):
                    continue
                if 'surface' in item:
                    item['surface'] = np.asarray(item['surface'], dtype=np.float32)
                if 'features' in item:
                    item['features'] = np.asarray(item['features'], dtype=np.float32)

        if self.mode == 'train':
            with open(self.path_cath + f'/cath42_pc_train_sorted.pkl', 'rb') as f:
                data_dict_cath = pickle.load(f)
            _downcast_float32_inplace(data_dict_cath)
            with open(self.path_afdb + f'/afdb-large4000.pkl', 'rb') as f:
                data_dict_afdb = pickle.load(f)
            _downcast_float32_inplace(data_dict_afdb)
            data_dict = {**data_dict_cath, **data_dict_afdb}
            return data_dict
        elif self.mode == 'valid':
            with open(self.path_cath + f'/cath42_pc_validation_sorted.pkl', 'rb') as f:
                data_dict = pickle.load(f)
            _downcast_float32_inplace(data_dict)
            return data_dict
        elif self.mode == 'test':
            with open(self.path_cath + f'/cath42_pc_test.pkl', 'rb') as f:
                data_dict = pickle.load(f)
            _downcast_float32_inplace(data_dict)
            return data_dict

    def change_mode(self, mode):
        self.mode = mode
        self.metadata = self._load_metadata()
        self.data_dict = self._load_data_dict()

    def __len__(self):
        return len(self.metadata)
    
    def _load_data_on_the_fly(self, index):
        entry = self.metadata[index]
        title = entry['title']
        seq_length = entry['seq_length']
        
        if title in self.data_dict:
            data = self.data_dict[title]
            data_entry = {
                'title': title,
                'seq': entry['seq'],
                'CA': entry['coords']['CA'],
                'C': entry['coords']['C'],
                'O': entry['coords']['O'],
                'N': entry['coords']['N'],
                'chain_mask': np.ones(seq_length),
                'chain_encoding': np.ones(seq_length),
                'orig_surface': data['surface'],
                'surface': normalize_coordinates(data['surface']),
                'features': data['features'][:, :2],
                # 'pc': data['pc'],
            }
            # ablation
            # data_entry['features'][:, 0] = np.random.rand(data['features'][:, 0].shape[0])
            # data_entry['features'][:, 1] = np.random.rand(data['features'][:, 1].shape[0])

            if self.mode == 'test':
                data_entry['category'] = 'Unknown'
                data_entry['score'] = 100.0
            
            return data_entry
        else:
            raise ValueError(f"Data for title {title} not found in the {self.mode} dictionary")

    def __getitem__(self, index):
        item = self._load_data_on_the_fly(index)
        L = len(item['seq'])
        if L > self.max_length:
            max_index = L - self.max_length
            truncate_index = random.randint(0, max_index)
            item['seq'] = item['seq'][truncate_index:truncate_index+self.max_length]
            item['CA'] = item['CA'][truncate_index:truncate_index+self.max_length]
            item['C'] = item['C'][truncate_index:truncate_index+self.max_length]
            item['O'] = item['O'][truncate_index:truncate_index+self.max_length]
            item['N'] = item['N'][truncate_index:truncate_index+self.max_length]
            item['chain_mask'] = item['chain_mask'][truncate_index:truncate_index+self.max_length]
            item['chain_encoding'] = item['chain_encoding'][truncate_index:truncate_index+self.max_length]
        return item