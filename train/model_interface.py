import inspect
import torch
import torch.nn as nn
import os
from src.interface.model_interface import MInterface_base
import math
from omegaconf import OmegaConf
from transformers.models.esm.openfold_utils.protein import to_pdb, Protein as OFProtein

# Define residue types (20 standard amino acids)
residue_types = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 
                 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

# Define a mapping from amino acid residues to indices (0 to 19)
residue_to_index = {
    'A': 0,  'R': 1,  'N': 2,  'D': 3,  'C': 4,
    'Q': 5,  'E': 6,  'G': 7,  'H': 8,  'I': 9,
    'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14,
    'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19
}


class MInterface(MInterface_base):
    def __init__(self, model_name=None, loss=None, lr=None, **kargs):
        super().__init__()
        self.save_hyperparameters()
        self.load_model()
        self.cross_entropy = nn.NLLLoss(reduction='none')
        os.makedirs(os.path.join(self.hparams.res_dir, self.hparams.ex_name), exist_ok=True)

    def forward(self, batch, mode='train', temperature=1.0):
        if self.hparams.augment_eps>0:
            batch['X'] = batch['X'] + self.hparams.augment_eps * torch.randn_like(batch['X'])

        batch = self.model._get_features(batch)
        results = self.model(batch)

        log_probs, mask = results['log_probs'], batch['mask']
        if len(log_probs.shape) == 3:
            loss = self.cross_entropy(log_probs.permute(0,2,1), batch['S'])
            loss = (loss*mask).sum()/(mask.sum())
        elif len(log_probs.shape) == 2:
            loss = self.cross_entropy(log_probs, batch['S'])
            
            loss = (loss*mask).sum()/(mask.sum())

        if self.hparams.model_name == 'SBC2Model':
            contrastive_loss = results['contrastive_loss']
            loss += contrastive_loss
        
        cmp = log_probs.argmax(dim=-1)==batch['S']
        recovery = (cmp*mask).sum()/(mask.sum())
        return loss, recovery

    def temperature_schedular(self, batch_idx):
        total_steps = self.hparams.steps_per_epoch*self.hparams.epoch
        
        initial_lr = 1.0
        circle_steps = total_steps//100
        x = batch_idx / total_steps
        threshold = 0.48
        if x<threshold:
            linear_decay = 1 - 2*x
        else:
            K = 1 - 2*threshold
            linear_decay = K - K*(x-threshold)/(1-threshold)
        
        new_lr = (1+math.cos(batch_idx/circle_steps*math.pi))/2*linear_decay*initial_lr

        return new_lr
    
    #https://lightning.ai/docs/pytorch/1.9.0/notebooks/lightning_examples/basic-gan.html
    def training_step(self, batch, batch_idx, **kwargs):
        loss, recovery = self(batch)
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    

    def validation_step(self, batch, batch_idx):
        loss, recovery = self(batch)
        self.log_dict({"val_loss":loss,
                       "recovery": recovery})
        
        return self.log_dict


    def test_step(self, batch, batch_idx):
        if not self.test_setupped:
            self.test_setup()   
            self.test_setupped = True 
        # Set model to evaluation mode and disable gradients
        self.model.eval()
        # losses, recoveries, plddt_cas, plddts, rmsds, tmscores, nssr_scores, hydrophobicity_rmsds, charge_rmsds = self.test_forward(batch)
        with torch.no_grad():
            # losses, recoveries, plddt_cas, plddts, rmsds, tmscores, nssr_scores, hydrophobicity_rmsds, charge_rmsds, surface_recoveries, core_recoveries = self.test_forward(batch)
            losses, recoveries, plddt_cas, plddts, rmsds, tmscores, nssr_scores, hydrophobicity_rmsds, charge_rmsds = self.test_forward(batch)

            for i, loss in enumerate(losses):
                recovery = recoveries[i]
                plddt_ca = plddt_cas[i]
                plddt = plddts[i]
                rmsd = rmsds[i]
                tmscore = tmscores[i]
                nssr_score = nssr_scores[i]
                hydrophobicity_rmsd = hydrophobicity_rmsds[i]
                charge_rmsd = charge_rmsds[i]
                # surface_recovery = surface_recoveries[i]
                # core_recovery = core_recoveries[i]
                self.test_step_outputs.append({
                        "test_loss": loss,
                        "test_recovery": recovery,
                        'test_plddt_ca': plddt_ca, 
                        'test_plddt': plddt,
                        'test_rmsd': rmsd,
                        'test_tmscore': tmscore,
                        'test_nssr_score': nssr_score,
                        'test_hydrophobicity_rmsd': hydrophobicity_rmsd,
                        'test_charge_rmsd': charge_rmsd,
                        # 'test_surface_recovery': surface_recovery,
                        # 'test_core_recovery': core_recovery,
                    }
                )

        return
        
    def load_model(self):
        params = OmegaConf.load(f'./src/models/configs/{self.hparams.model_name}.yaml')
        params.update(self.hparams)
        if self.hparams.model_name == 'SBC2Model':
            from src.models.SBC2_model import SBC2Model
            self.model = SBC2Model(params)

    def instancialize(self, Model, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.hparams.
        """
        class_args = inspect.getargspec(Model.__init__).args[1:]
        inkeys = self.hparams.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = getattr(self.hparams, arg)
        args1.update(other_args)
        return Model(**args1)
