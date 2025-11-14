import pytorch_lightning as pl


class DInterface_base(pl.LightningDataModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.batch_size = self.hparams.batch_size
        print("batch_size", self.batch_size)
        self.load_data_module()
