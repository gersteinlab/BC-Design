import inspect
from torch.utils.data import DataLoader
from src.interface.data_interface import DInterface_base
import torch
import os.path as osp
from src.tools.utils import cuda

class MyDataLoader(DataLoader):
    def __init__(self, dataset, model_name, batch_size=64, num_workers=8, *args, **kwargs):
        super().__init__(dataset, batch_size=batch_size, num_workers=num_workers, *args, **kwargs)
        self.pretrain_device = 'cuda:0'
        self.model_name = model_name
    
    def __iter__(self):
        for batch in super().__iter__():
            try:
                self.pretrain_device = f'cuda:{torch.distributed.get_rank()}'
            except:
                self.pretrain_device = 'cuda:0'

            stream = torch.cuda.Stream(
                self.pretrain_device
            )
            with torch.cuda.stream(stream):
                if self.model_name=='GVP':
                    batch = batch.cuda(non_blocking=True, device=self.pretrain_device)
                    yield batch
                else:
                    for key, val in batch.items():
                        if type(val) == torch.Tensor:
                            batch[key] = batch[key].cuda(non_blocking=True, device=self.pretrain_device)

                    yield batch


class DInterface(DInterface_base):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters()
        self.load_data_module()

    def setup(self, stage=None):
        from src.datasets.featurizer import (featurize_SBC2Model)
        if self.hparams.model_name == 'SBC2Model':
            self.collate_fn = featurize_SBC2Model
    
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.trainset = self.instancialize(split = 'train')
            self.valset = self.instancialize(split='valid')
    
        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.testset = self.instancialize(split='test')

    def train_dataloader(self):
        return MyDataLoader(self.trainset, model_name=self.hparams.model_name, batch_size=self.batch_size, num_workers=self.hparams.num_workers, shuffle=True, prefetch_factor=None, pin_memory=True, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return MyDataLoader(self.valset, model_name=self.hparams.model_name, batch_size=self.batch_size, num_workers=self.hparams.num_workers, shuffle=False, prefetch_factor=None, pin_memory=True, collate_fn=self.collate_fn)

    def test_dataloader(self):
        return MyDataLoader(self.testset, model_name=self.hparams.model_name, batch_size=self.batch_size, num_workers=self.hparams.num_workers, shuffle=False, prefetch_factor=None, pin_memory=True, collate_fn=self.collate_fn)

    def load_data_module(self):
        name = self.hparams.dataset
        
        if name == 'CATH4.2':
            from src.datasets.cath_dataset import CATHDataset
            self.data_module = CATHDataset
            self.hparams['version'] = 4.2
            self.hparams['path'] = osp.join(self.hparams.data_root, 'cath4.2')

        if name == 'TS50':
            from src.datasets.ts_dataset  import TS50Dataset
            self.data_module = TS50Dataset
            self.hparams['path'] = osp.join(self.hparams.data_root, 'ts50')

        if name == 'TS500':
            from src.datasets.ts_dataset  import TS500Dataset
            self.data_module = TS500Dataset
            self.hparams['path'] = osp.join(self.hparams.data_root, 'ts500')

        if name == 'AFDB2000':
            from src.datasets.afdb_dataset  import AFDB2000Dataset
            self.data_module = AFDB2000Dataset
            self.hparams['path'] = osp.join(self.hparams.data_root, 'afdb2000')

    def instancialize(self, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.kwargs.
        """
        
        class_args =  list(inspect.signature(self.data_module.__init__).parameters)[1:]
        inkeys = self.hparams.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = self.hparams[arg]
        args1.update(other_args)
        return self.data_module(**args1)