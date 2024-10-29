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


class CATHDataset(data.Dataset):
    def __init__(self, path='./', split='train', max_length=500, test_name='All', data=None, removeTS=0, version=4.2):
        self.version = version
        self.path = path
        self.mode = split
        self.max_length = max_length
        self.test_name = test_name
        self.removeTS = removeTS
        
        if self.removeTS:
            self.remove = json.load(open(self.path + '/remove.json', 'r'))['remove']
        
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
        with open(self.path + '/chain_set_splits.json') as f:
            dataset_splits = json.load(f)
        
        # Handle specific test splits if needed
        if self.test_name == 'L100': 
            with open(self.path + '/test_split_L100.json') as f:
                test_splits = json.load(f)
            dataset_splits['test'] = test_splits['test']

        if self.test_name == 'sc': 
            with open(self.path + '/test_split_sc.json') as f:
                test_splits = json.load(f)
            dataset_splits['test'] = test_splits['test']

        # Select the appropriate split
        if self.mode == 'valid':
            valid_titles = set(dataset_splits['validation'])
        else:
            valid_titles = set(dataset_splits[self.mode])

        if not os.path.exists(self.path):
            raise FileNotFoundError("No such file: {} !!!".format(self.path))
        else:
            with open(self.path + '/chain_set.jsonl') as f:
                lines = f.readlines()
            for line in tqdm(lines):
                entry = json.loads(line)
                if self.removeTS and entry['name'] in self.remove:
                    continue
                
                bad_chars = set([s for s in entry['seq']]).difference(alphabet_set)
                if len(bad_chars) == 0 and len(entry['seq']) <= self.max_length and entry['name'] in valid_titles:
                    metadata.append({
                        'title': entry['name'],
                        'seq_length': len(entry['seq']),
                        'seq': entry['seq'],
                        'coords': entry['coords'],
                    })
        return metadata

    def _load_data_dict(self):
        # Load the appropriate pickle file based on the mode and keep it in memory
        if self.mode == 'train':
            with open(self.path + f'/cath42_pc_train_sorted.pkl', 'rb') as f:
                return pickle.load(f)
        elif self.mode == 'valid':
            with open(self.path + f'/cath42_pc_validation_sorted.pkl', 'rb') as f:
                return pickle.load(f)
        elif self.mode == 'test':
            with open(self.path + f'/cath42_pc_test.pkl', 'rb') as f:
                return pickle.load(f)

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
                'CA': np.asarray(entry['coords']['CA']),
                'C': np.asarray(entry['coords']['C']),
                'O': np.asarray(entry['coords']['O']),
                'N': np.asarray(entry['coords']['N']),
                'chain_mask': np.ones(seq_length),
                'chain_encoding': np.ones(seq_length),
                'orig_surface': data['surface'],
                'surface': normalize_coordinates(data['surface']),
                'features': data['features'][:, :2],
            }
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
