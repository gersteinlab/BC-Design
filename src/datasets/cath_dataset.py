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


class CATHDataset(data.Dataset):
    def __init__(self, path='./',  split='train', max_length=500, test_name='All', data = None, removeTS=0, version=4.2):
        self.version = version
        self.path = path
        self.mode = split
        self.max_length = max_length
        self.test_name = test_name
        self.removeTS = removeTS
        if self.removeTS:
            self.remove = json.load(open(self.path+'/remove.json', 'r'))['remove']
        
        if data is None:
            self.data = self.cache_data[split]
        else:
            self.data = data
        
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D", cache_dir="gaozhangyang/model_zoom/transformers")
    
    @cached_property
    def cache_data(self):
        alphabet='ACDEFGHIKLMNPQRSTVWY'
        alphabet_set = set([a for a in alphabet])
        if not os.path.exists(self.path):
            raise FileNotFoundError("no such file: {} !!!".format(self.path))
        else:
            with open(self.path+'/chain_set.jsonl') as f:
                lines = f.readlines()
            data_list = []
            for line in tqdm(lines):
                entry = json.loads(line)
                if self.removeTS and entry['name'] in self.remove:
                    continue
                seq = entry['seq']

                for key, val in entry['coords'].items():
                    entry['coords'][key] = np.asarray(val)
                
                bad_chars = set([s for s in seq]).difference(alphabet_set)

                if len(bad_chars) == 0:
                    if len(entry['seq']) <= self.max_length: 
                        chain_length = len(entry['seq'])
                        chain_mask = np.ones(chain_length)
                        data_list.append({
                            'title':entry['name'],
                            'seq':entry['seq'],
                            'CA':entry['coords']['CA'],
                            'C':entry['coords']['C'],
                            'O':entry['coords']['O'],
                            'N':entry['coords']['N'],
                            'chain_mask': chain_mask,
                            'chain_encoding': 1*chain_mask
                        })
                        
            if self.version==4.2:
                with open(self.path+'/chain_set_splits.json') as f:
                    dataset_splits = json.load(f)
            
            if self.version==4.3:
                with open(self.path+'/chain_set_splits.json') as f:
                    dataset_splits = json.load(f)
            
            
            if self.test_name == 'L100': 
                with open(self.path+'/test_split_L100.json') as f:
                    test_splits = json.load(f)
                dataset_splits['test'] = test_splits['test']

            if self.test_name == 'sc':
                with open(self.path+'/test_split_sc.json') as f:
                    test_splits = json.load(f)
                dataset_splits['test'] = test_splits['test']
            
            name2set = {}
            name2set.update({name:'train' for name in dataset_splits['train']})
            name2set.update({name:'valid' for name in dataset_splits['validation']})
            name2set.update({name:'test' for name in dataset_splits['test']})

            data_dict = {'train':[],'valid':[],'test':[]}
            for data in data_list:
                if name2set.get(data['title']):
                    if name2set[data['title']] == 'train':
                        data_dict['train'].append(data)
                    
                    if name2set[data['title']] == 'valid':
                        data_dict['valid'].append(data)
                    
                    if name2set[data['title']] == 'test':
                        data['category'] = 'Unkown'
                        data['score'] = 100.0
                        data_dict['test'].append(data)
            return data_dict

    def change_mode(self, mode):
        self.data = self.cache_data[mode]
    
    def __len__(self):
        return len(self.data)
    
    def get_item(self, index):
        return self.data[index]
    
    def __getitem__(self, index):
        item = self.data[index]
        L = len(item['seq'])
        if L>self.max_length:
            # 计算截断的最大索引
            max_index = L - self.max_length
            # 生成随机的截断索引
            truncate_index = random.randint(0, max_index)
            # 进行截断
            item['seq'] = item['seq'][truncate_index:truncate_index+self.max_length]
            item['CA'] = item['CA'][truncate_index:truncate_index+self.max_length]
            item['C'] = item['C'][truncate_index:truncate_index+self.max_length]
            item['O'] = item['O'][truncate_index:truncate_index+self.max_length]
            item['N'] = item['N'][truncate_index:truncate_index+self.max_length]
            item['chain_mask'] = item['chain_mask'][truncate_index:truncate_index+self.max_length]
            item['chain_encoding'] = item['chain_encoding'][truncate_index:truncate_index+self.max_length]
        return item


class CATHDatasetSurfProPiFold(data.Dataset):
    def __init__(self, path='./',  split='train', max_length=500, test_name='All', data = None, removeTS=0, version=4.2):
        self.version = version
        self.path = path
        self.mode = split
        self.max_length = max_length
        self.test_name = test_name
        self.removeTS = removeTS
        if self.removeTS:
            self.remove = json.load(open(self.path+'/remove.json', 'r'))['remove']
        
        if data is None:
            self.data = self.cache_data[split]
        else:
            self.data = data
        
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D", cache_dir="gaozhangyang/model_zoom/transformers")
    
    @cached_property
    def cache_data(self):
        alphabet='ACDEFGHIKLMNPQRSTVWY'
        alphabet_set = set([a for a in alphabet])
        if not os.path.exists(self.path):
            raise FileNotFoundError("no such file: {} !!!".format(self.path))
        else:
            with open(self.path+'/chain_set.jsonl') as f:
                lines = f.readlines()
            data_list = []
            for line in tqdm(lines):
                entry = json.loads(line)
                if self.removeTS and entry['name'] in self.remove:
                    continue
                seq = entry['seq']

                for key, val in entry['coords'].items():
                    entry['coords'][key] = np.asarray(val)
                
                bad_chars = set([s for s in seq]).difference(alphabet_set)

                if len(bad_chars) == 0:
                    if len(entry['seq']) <= self.max_length: 
                        chain_length = len(entry['seq'])
                        chain_mask = np.ones(chain_length)
                        data_list.append({
                            'title':entry['name'],
                            'seq':entry['seq'],
                            'CA':entry['coords']['CA'],
                            'C':entry['coords']['C'],
                            'O':entry['coords']['O'],
                            'N':entry['coords']['N'],
                            'chain_mask': chain_mask,
                            'chain_encoding': 1*chain_mask
                        })
            
            if self.version==4.2:
                with open(self.path+'/chain_set_splits.json') as f:
                    dataset_splits = json.load(f)
            
            if self.version==4.3:
                with open(self.path+'/chain_set_splits.json') as f:
                    dataset_splits = json.load(f)
            
            if self.test_name == 'L100': 
                with open(self.path+'/test_split_L100.json') as f:
                    test_splits = json.load(f)
                dataset_splits['test'] = test_splits['test']

            if self.test_name == 'sc': 
                with open(self.path+'/test_split_sc.json') as f:
                    test_splits = json.load(f)
                dataset_splits['test'] = test_splits['test']
            
            name2set = {}
            name2set.update({name:'train' for name in dataset_splits['train']})
            name2set.update({name:'valid' for name in dataset_splits['validation']})
            name2set.update({name:'test' for name in dataset_splits['test']})

            # Load pickle files
            with open(self.path + '/cath42_pc_train_sorted.pkl', 'rb') as f:
                train_dict = pickle.load(f)
            with open(self.path + '/cath42_pc_validation_sorted.pkl', 'rb') as f:
                validation_dict = pickle.load(f)
            with open(self.path + '/cath42_pc_test.pkl', 'rb') as f:
                test_dict = pickle.load(f)

            # Combine data from JSONL and pickle files
            data_dict = {'train': [], 'valid': [], 'test': []}
            for data in data_list:
                title = data['title']
                if title in name2set:
                    data_type = name2set[title]
                    if data_type == 'train' and title in train_dict:
                        data['orig_surface'] = train_dict[title]['surface']
                        data['surface'] = normalize_coordinates(train_dict[title]['surface'])
                        data['features'] = train_dict[title]['features'][:, :2]
                        data['pc'] = train_dict[title]['pc']
                        data_dict['train'].append(data)
                    elif data_type == 'valid' and title in validation_dict:
                        data['orig_surface'] = validation_dict[title]['surface']
                        data['surface'] = normalize_coordinates(validation_dict[title]['surface'])
                        data['features'] = validation_dict[title]['features'][:, :2]
                        data['pc'] = validation_dict[title]['pc']
                        data_dict['valid'].append(data)
                    elif data_type == 'test' and title in test_dict:
                        data['orig_surface'] = test_dict[title]['surface']
                        data['surface'] = normalize_coordinates(test_dict[title]['surface'])
                        data['features'] = test_dict[title]['features'][:, :2]
                        data['pc'] = test_dict[title]['pc']
                        data['category'] = 'Unknown'
                        data['score'] = 100.0
                        data_dict['test'].append(data)

            # max_float = np.finfo(np.float64).max
            # for data in tqdm(data_dict['train'] + data_dict['valid'] + data_dict['test']):
            #     surface_coords = data['surface']
            #     ca_coords = data['CA']
            #     # nbrs = NearestNeighbors(n_neighbors=8, algorithm='ball_tree').fit(np.nan_to_num(ca_coords, nan=max_float))
            #     # distances, indices = nbrs.kneighbors(surface_coords)
            #     # ss_connection = np.zeros((ca_coords.shape[0], surface_coords.shape[0]))
            #     # for i, neighbors in enumerate(indices):
            #     #     ss_connection[neighbors, i] = 1

            #     # 找到非NaN的ca_coords点
            #     valid_indices = ~np.isnan(ca_coords).any(axis=1)
            #     valid_ca_coords = ca_coords[valid_indices]
                
            #     # 将 surface_coords 作为训练点，valid_ca_coords 作为查询点
            #     nbrs = NearestNeighbors(n_neighbors=8, algorithm='ball_tree').fit(surface_coords)
            #     distances, indices = nbrs.kneighbors(valid_ca_coords)
                
            #     ss_connection = np.zeros((ca_coords.shape[0], surface_coords.shape[0]))
                
            #     # 处理非NaN的ca_coords点
            #     for i, neighbors in zip(np.where(valid_indices)[0], indices):
            #         ss_connection[i, neighbors] = 1
                
            #     # 处理NaN的ca_coords点
            #     ss_connection[~valid_indices, :] = 1
            #     data['ss_connection'] = ss_connection

            return data_dict
    # @cached_property
    # def cache_data(self):
    #     alphabet='ACDEFGHIKLMNPQRSTVWY'
    #     alphabet_set = set([a for a in alphabet])
    #     if not os.path.exists(self.path):
    #         raise FileNotFoundError("no such file: {} !!!".format(self.path))
    #     else:
    #         with open(self.path+'/chain_set.jsonl') as f:
    #             lines = f.readlines()
    #         data_list = []
    #         for line in tqdm(lines):
    #             entry = json.loads(line)
    #             if self.removeTS and entry['name'] in self.remove:
    #                 continue
    #             seq = entry['seq']

    #             for key, val in entry['coords'].items():
    #                 entry['coords'][key] = np.asarray(val)
                
    #             bad_chars = set([s for s in seq]).difference(alphabet_set)

    #             if len(bad_chars) == 0:
    #                 if len(entry['seq']) <= self.max_length: 
    #                     chain_length = len(entry['seq'])
    #                     chain_mask = np.ones(chain_length)
    #                     data_list.append({
    #                         'title':entry['name'],
    #                         'seq':entry['seq'],
    #                         'CA':entry['coords']['CA'],
    #                         'C':entry['coords']['C'],
    #                         'O':entry['coords']['O'],
    #                         'N':entry['coords']['N'],
    #                         'chain_mask': chain_mask,
    #                         'chain_encoding': 1*chain_mask
    #                     })
            
    #         if self.version==4.2:
    #             with open(self.path+'/chain_set_splits.json') as f:
    #                 dataset_splits = json.load(f)
            
    #         if self.version==4.3:
    #             with open(self.path+'/chain_set_splits.json') as f:
    #                 dataset_splits = json.load(f)
            
    #         if self.test_name == 'L100': 
    #             with open(self.path+'/test_split_L100.json') as f:
    #                 test_splits = json.load(f)
    #             dataset_splits['test'] = test_splits['test']

    #         if self.test_name == 'sc': 
    #             with open(self.path+'/test_split_sc.json') as f:
    #                 test_splits = json.load(f)
    #             dataset_splits['test'] = test_splits['test']
            
    #         name2set = {}
    #         name2set.update({name:'train' for name in dataset_splits['train']})
    #         name2set.update({name:'valid' for name in dataset_splits['validation']})
    #         name2set.update({name:'test' for name in dataset_splits['test']})

    #         # Load pickle files
    #         with open(self.path + '/cath42_pc_train_sorted.pkl', 'rb') as f:
    #             train_dict = pickle.load(f)
    #         with open(self.path + '/cath42_pc_validation_sorted.pkl', 'rb') as f:
    #             validation_dict = pickle.load(f)
    #         with open(self.path + '/cath42_pc_test.pkl', 'rb') as f:
    #             test_dict = pickle.load(f)

    #         # Combine data from JSONL and pickle files
    #         data_dict = {'train': [], 'valid': [], 'test': []}
    #         for data in data_list:
    #             title = data['title']
    #             if title in name2set:
    #                 data_type = name2set[title]
    #                 if data_type == 'train' and title in train_dict:
    #                     data['surface'] = normalize_coordinates(train_dict[title]['surface'])
    #                     data['features'] = train_dict[title]['features'][:, :2]
    #                     data['pc'] = train_dict[title]['pc']
    #                     data['ss_connection'] = train_dict[title]['ss_connection']
    #                     data_dict['train'].append(data)
    #                 elif data_type == 'valid' and title in validation_dict:
    #                     data['surface'] = normalize_coordinates(validation_dict[title]['surface'])
    #                     data['features'] = validation_dict[title]['features'][:, :2]
    #                     data['pc'] = validation_dict[title]['pc']
    #                     data['ss_connection'] = validation_dict[title]['ss_connection']
    #                     data_dict['valid'].append(data)
    #                 elif data_type == 'test' and title in test_dict:
    #                     data['surface'] = normalize_coordinates(test_dict[title]['surface'])
    #                     data['features'] = test_dict[title]['features'][:, :2]
    #                     data['pc'] = test_dict[title]['pc']
    #                     data['ss_connection'] = test_dict[title]['ss_connection']
    #                     data['category'] = 'Unknown'
    #                     data['score'] = 100.0
    #                     data_dict['test'].append(data)

    #         # Free memory
    #         del lines, data_list, dataset_splits
    #         del name2set, train_dict, validation_dict, test_dict
    #         gc.collect()

    #         return data_dict

    def change_mode(self, mode):
        self.data = self.cache_data[mode]
    
    def __len__(self):
        return len(self.data)
    
    def get_item(self, index):
        return self.data[index]
    
    def __getitem__(self, index):
        item = self.data[index]
        L = len(item['seq'])
        if L>self.max_length:
            # 计算截断的最大索引
            max_index = L - self.max_length
            # 生成随机的截断索引
            truncate_index = random.randint(0, max_index)
            # 进行截断
            item['seq'] = item['seq'][truncate_index:truncate_index+self.max_length]
            item['CA'] = item['CA'][truncate_index:truncate_index+self.max_length]
            item['C'] = item['C'][truncate_index:truncate_index+self.max_length]
            item['O'] = item['O'][truncate_index:truncate_index+self.max_length]
            item['N'] = item['N'][truncate_index:truncate_index+self.max_length]
            item['chain_mask'] = item['chain_mask'][truncate_index:truncate_index+self.max_length]
            item['chain_encoding'] = item['chain_encoding'][truncate_index:truncate_index+self.max_length]
            item['ss_connection'] = item['ss_connection'][truncate_index:truncate_index + self.max_length, :]
        return item


class CATHDatasetSurfProPiFoldDense(data.Dataset):
    def __init__(self, path='./',  split='train', max_length=500, test_name='All', data = None, removeTS=0, version=4.2):
        self.version = version
        self.path = path
        self.mode = split
        self.max_length = max_length
        self.test_name = test_name
        self.removeTS = removeTS
        if self.removeTS:
            self.remove = json.load(open(self.path+'/remove.json', 'r'))['remove']
        
        if data is None:
            self.data = self.cache_data[split]
        else:
            self.data = data
        
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D", cache_dir="gaozhangyang/model_zoom/transformers")
    
    @cached_property
    def cache_data(self):
        alphabet='ACDEFGHIKLMNPQRSTVWY'
        alphabet_set = set([a for a in alphabet])
        if not os.path.exists(self.path):
            raise FileNotFoundError("no such file: {} !!!".format(self.path))
        else:
            with open(self.path+'/chain_set.jsonl') as f:
                lines = f.readlines()
            data_list = []
            for line in tqdm(lines):
                entry = json.loads(line)
                if self.removeTS and entry['name'] in self.remove:
                    continue
                seq = entry['seq']

                for key, val in entry['coords'].items():
                    entry['coords'][key] = np.asarray(val)
                
                bad_chars = set([s for s in seq]).difference(alphabet_set)

                if len(bad_chars) == 0:
                    if len(entry['seq']) <= self.max_length: 
                        chain_length = len(entry['seq'])
                        chain_mask = np.ones(chain_length)
                        data_list.append({
                            'title':entry['name'],
                            'seq':entry['seq'],
                            'CA':entry['coords']['CA'],
                            'C':entry['coords']['C'],
                            'O':entry['coords']['O'],
                            'N':entry['coords']['N'],
                            'chain_mask': chain_mask,
                            'chain_encoding': 1*chain_mask
                        })
            
            if self.version==4.2:
                with open(self.path+'/chain_set_splits.json') as f:
                    dataset_splits = json.load(f)
            
            if self.version==4.3:
                with open(self.path+'/chain_set_splits.json') as f:
                    dataset_splits = json.load(f)
            
            if self.test_name == 'L100': 
                with open(self.path+'/test_split_L100.json') as f:
                    test_splits = json.load(f)
                dataset_splits['test'] = test_splits['test']

            if self.test_name == 'sc': 
                with open(self.path+'/test_split_sc.json') as f:
                    test_splits = json.load(f)
                dataset_splits['test'] = test_splits['test']
            
            name2set = {}
            name2set.update({name:'train' for name in dataset_splits['train']})
            name2set.update({name:'valid' for name in dataset_splits['validation']})
            name2set.update({name:'test' for name in dataset_splits['test']})

            # Load pickle files
            with open(self.path + '/cath42_pc_train_sorted.pkl', 'rb') as f:
                train_dict = pickle.load(f)
                # train_dict = sample_if_needed(train_dict)
            with open(self.path + '/cath42_pc_validation_sorted.pkl', 'rb') as f:
                validation_dict = pickle.load(f)
                # validation_dict = sample_if_needed(validation_dict)
            with open(self.path + '/cath42_pc_test.pkl', 'rb') as f:
                test_dict = pickle.load(f)
                # test_dict = sample_if_needed(test_dict)

            # Combine data from JSONL and pickle files
            data_dict = {'train': [], 'valid': [], 'test': []}
            n_node_thr = 4000
            for data in tqdm(data_list):
                title = data['title']
                if title in name2set:
                    data_type = name2set[title]
                    if data_type == 'train' and title in train_dict:
                        data['orig_surface'] = train_dict[title]['surface']
                        data['surface'] = normalize_coordinates(train_dict[title]['surface'])
                        data['features'] = train_dict[title]['features'][:, :2]
                        data['pc'] = train_dict[title]['pc']
                        data_dict['train'].append(data)
                    elif data_type == 'valid' and title in validation_dict:
                        data['orig_surface'] = validation_dict[title]['surface']
                        data['surface'] = normalize_coordinates(validation_dict[title]['surface'])
                        data['features'] = validation_dict[title]['features'][:, :2]
                        data['pc'] = validation_dict[title]['pc']
                        data_dict['valid'].append(data)
                    elif data_type == 'test' and title in test_dict:
                        data['orig_surface'] = test_dict[title]['surface']
                        data['surface'] = normalize_coordinates(test_dict[title]['surface'])
                        data['features'] = test_dict[title]['features'][:, :2]
                        data['pc'] = test_dict[title]['pc']
                        data['category'] = 'Unknown'
                        data['score'] = 100.0
                        data_dict['test'].append(data)

            return data_dict

    def change_mode(self, mode):
        self.data = self.cache_data[mode]
    
    def __len__(self):
        return len(self.data)
    
    def get_item(self, index):
        return self.data[index]
    
    def __getitem__(self, index):
        item = self.data[index]
        L = len(item['seq'])
        if L>self.max_length:
            # 计算截断的最大索引
            max_index = L - self.max_length
            # 生成随机的截断索引
            truncate_index = random.randint(0, max_index)
            # 进行截断
            item['seq'] = item['seq'][truncate_index:truncate_index+self.max_length]
            item['CA'] = item['CA'][truncate_index:truncate_index+self.max_length]
            item['C'] = item['C'][truncate_index:truncate_index+self.max_length]
            item['O'] = item['O'][truncate_index:truncate_index+self.max_length]
            item['N'] = item['N'][truncate_index:truncate_index+self.max_length]
            item['chain_mask'] = item['chain_mask'][truncate_index:truncate_index+self.max_length]
            item['chain_encoding'] = item['chain_encoding'][truncate_index:truncate_index+self.max_length]
            item['ss_connection'] = item['ss_connection'][truncate_index:truncate_index + self.max_length, :]
        return item


# class CATHDatasetSurfProPiFoldDenseLarge(data.Dataset):
#     def __init__(self, path='./', split='train', max_length=500, test_name='All', data=None, removeTS=0, version=4.2):
#         self.version = version
#         self.path = path
#         self.mode = split
#         self.max_length = max_length
#         self.test_name = test_name
#         self.removeTS = removeTS
        
#         if self.removeTS:
#             self.remove = json.load(open(self.path + '/remove.json', 'r'))['remove']
        
#         if data is None:
#             self.metadata = self._load_metadata()
#         else:
#             self.metadata = data
        
#         self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D", cache_dir="gaozhangyang/model_zoom/transformers")
        
#         # Load the entire dictionary corresponding to the current mode
#         self.data_dict = self._load_data_dict()
#     def _load_metadata(self):
#         alphabet = 'ACDEFGHIKLMNPQRSTVWY'
#         alphabet_set = set([a for a in alphabet])
#         metadata = []
        
#         # Load the split JSON files
#         with open(self.path + '/chain_set_splits.json') as f:
#             dataset_splits = json.load(f)
        
#         # Handle specific test splits if needed
#         if self.test_name == 'L100': 
#             with open(self.path + '/test_split_L100.json') as f:
#                 test_splits = json.load(f)
#             dataset_splits['test'] = test_splits['test']

#         if self.test_name == 'sc': 
#             with open(self.path + '/test_split_sc.json') as f:
#                 test_splits = json.load(f)
#             dataset_splits['test'] = test_splits['test']

#         # Select the appropriate split
#         if self.mode == 'valid':
#             valid_titles = set(dataset_splits['validation'])
#         else:
#             valid_titles = set(dataset_splits[self.mode])

#         if not os.path.exists(self.path):
#             raise FileNotFoundError("No such file: {} !!!".format(self.path))
#         else:
#             with open(self.path + '/chain_set.jsonl') as f:
#                 lines = f.readlines()
#             for line in tqdm(lines):
#                 entry = json.loads(line)
#                 if self.removeTS and entry['name'] in self.remove:
#                     continue
                
#                 bad_chars = set([s for s in entry['seq']]).difference(alphabet_set)
#                 if len(bad_chars) == 0 and len(entry['seq']) <= self.max_length and entry['name'] in valid_titles:
#                     metadata.append({
#                         'title': entry['name'],
#                         'seq_length': len(entry['seq']),
#                         'seq': entry['seq'],
#                         'coords': entry['coords'],
#                     })
#         return metadata

#     def _load_data_dict(self):
#         # Load the appropriate pickle file based on the mode and keep it in memory
#         if self.mode == 'train':
#             with open(self.path + f'/cath42_pc_train_sorted.pkl', 'rb') as f:
#                 return pickle.load(f)
#         elif self.mode == 'valid':
#             with open(self.path + f'/cath42_pc_validation_sorted.pkl', 'rb') as f:
#                 return pickle.load(f)
#         elif self.mode == 'test':
#             with open(self.path + f'/cath42_pc_test.pkl', 'rb') as f:
#                 return pickle.load(f)

#     def change_mode(self, mode):
#         self.mode = mode
#         self.metadata = self._load_metadata()
#         self.data_dict = self._load_data_dict()

#     def __len__(self):
#         return len(self.metadata)
    
#     def _load_data_on_the_fly(self, index):
#         entry = self.metadata[index]
#         title = entry['title']
#         seq_length = entry['seq_length']
        
#         if title in self.data_dict:
#             data = self.data_dict[title]
#             data_entry = {
#                 'title': title,
#                 'seq': entry['seq'],
#                 'CA': np.asarray(entry['coords']['CA']),
#                 'C': np.asarray(entry['coords']['C']),
#                 'O': np.asarray(entry['coords']['O']),
#                 'N': np.asarray(entry['coords']['N']),
#                 'chain_mask': np.ones(seq_length),
#                 'chain_encoding': np.ones(seq_length),
#                 'orig_surface': data['surface'],
#                 'surface': normalize_coordinates(data['surface']),
#                 'features': data['features'][:, :2],
#                 # 'pc': data['pc'],
#             }
#             # ablation
#             # data_entry['features'][:, 0] = np.random.rand(data['features'][:, 0].shape[0])
#             # data_entry['features'][:, 1] = np.random.rand(data['features'][:, 1].shape[0])

#             if self.mode == 'test':
#                 data_entry['category'] = 'Unknown'
#                 data_entry['score'] = 100.0
            
#             return data_entry
#         else:
#             raise ValueError(f"Data for title {title} not found in the {self.mode} dictionary")

#     def __getitem__(self, index):
#         item = self._load_data_on_the_fly(index)
#         L = len(item['seq'])
#         if L > self.max_length:
#             max_index = L - self.max_length
#             truncate_index = random.randint(0, max_index)
#             item['seq'] = item['seq'][truncate_index:truncate_index+self.max_length]
#             item['CA'] = item['CA'][truncate_index:truncate_index+self.max_length]
#             item['C'] = item['C'][truncate_index:truncate_index+self.max_length]
#             item['O'] = item['O'][truncate_index:truncate_index+self.max_length]
#             item['N'] = item['N'][truncate_index:truncate_index+self.max_length]
#             item['chain_mask'] = item['chain_mask'][truncate_index:truncate_index+self.max_length]
#             item['chain_encoding'] = item['chain_encoding'][truncate_index:truncate_index+self.max_length]
#         return item

class CATHDatasetSurfProPiFoldDenseLarge(data.Dataset):
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


