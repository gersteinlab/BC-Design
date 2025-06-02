import inspect
import torch
import torch.nn as nn
import os
from torcheval.metrics.text import Perplexity
from src.interface.model_interface import MInterface_base
import math
from omegaconf import OmegaConf
from transformers import AutoTokenizer, EsmForProteinFolding
from transformers.models.esm.openfold_utils.protein import to_pdb, Protein as OFProtein
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37
from Bio import PDB
from Bio.PDB import Superimposer, Structure, Model, Chain, Residue
from Bio.PDB.Atom import Atom
from Bio.Align import substitution_matrices
from io import StringIO
import subprocess
import time
import pandas as pd
import statistics

tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D", cache_dir="gaozhangyang/model_zoom/transformers") # mask token: 32
esmfold_tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1", cache_dir="cache/transformers/tokenizers")

# Load BLOSUM62 substitution matrix
blosum62 = substitution_matrices.load("BLOSUM62")


def calculate_tm_score(pred_pdb_file, gt_pdb_file):
    """
    Calculate TM-score using the external TM-align tool.
    :param pred_pdb_file: Path to predicted PDB file
    :param gt_pdb_file: Path to ground truth PDB file
    :return: TM-score as a float
    """
    # Command to run TM-align
    command = f"./TMscore {pred_pdb_file} {gt_pdb_file}"
    
    # Execute TM-align using subprocess and capture output
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    stdout, stderr = process.communicate()

    # Check for errors
    if process.returncode != 0:
        raise RuntimeError(f"TM-align failed: {stderr.decode('utf-8')}")

    # Parse the TM-score from TM-align output
    output = stdout.decode('utf-8')
    tm_score = None

    for line in output.split('\n'):
        if "TM-score    =" in line:
            # Extract the TM-score value from the output
            tm_score = float(line.split('=')[1].split()[0])
            break

    if tm_score is None:
        raise ValueError("TM-score not found in TM-align output")

    return tm_score


def convert_outputs_to_pdb(outputs):
    final_atom_positions = atom14_to_atom37(outputs["positions"][-1], outputs)
    outputs = {k: v.to("cpu").numpy() for k, v in outputs.items()}
    final_atom_positions = final_atom_positions.cpu().numpy()
    final_atom_mask = outputs["atom37_atom_exists"]
    pdbs = []
    for i in range(outputs["aatype"].shape[0]):
        aa = outputs["aatype"][i]
        pred_pos = final_atom_positions[i]
        mask = final_atom_mask[i]
        resid = outputs["residue_index"][i] + 1
        pred = OFProtein(
            aatype=aa,
            atom_positions=pred_pos,
            atom_mask=mask,
            residue_index=resid,
            b_factors=outputs["plddt"][i],
            chain_index=outputs["chain_index"][i] if "chain_index" in outputs else None,
        )
        pdbs.append(to_pdb(pred))
    return pdbs


def calculate_average_plddt(pdb_content):
    # Use StringIO to convert the PDB content (string) into a file-like object
    pdb_io = StringIO(pdb_content)
    parser = PDB.PDBParser(QUIET=True)
    
    # Parse the structure from the file-like object
    structure = parser.get_structure('structure', pdb_io)
    
    b_factors = []
    b_factors_ca = []
    
    # Iterate over all atoms in the structure
    for model in structure:
        for chain in model:
            for residue in chain:
                # Check if the residue contains a Cα atom
                if 'CA' in residue:
                    ca_atom = residue['CA']
                    # Extract the B-factor (pLDDT in your case) from the Cα atom
                    b_factor_ca = ca_atom.get_bfactor()
                    b_factors_ca.append(b_factor_ca)
                for atom in residue:
                    # Extract the B-factor (pLDDT in your case)
                    b_factor = atom.get_bfactor()
                    b_factors.append(b_factor)
    
    if b_factors_ca:
        # Calculate the average pLDDT for Cα atoms
        plddt_ca = sum(b_factors_ca) / len(b_factors_ca)
    else:
        plddt_ca = None

    if b_factors:
        # Calculate the average pLDDT for Cα atoms
        plddt = sum(b_factors) / len(b_factors)
    else:
        plddt = None   

    return {'plddt_ca': plddt_ca, 'plddt': plddt}


def save_pdbs_to_files(pdbs, titles, directory):
    # Ensure the directory exists
    os.makedirs(directory, exist_ok=True)

    for i, pdb in enumerate(pdbs):
        pdb_filename = os.path.join(directory, f'{titles[i]}.pdb')
    #     with open(pdb_filename, 'w') as pdb_file:
    #         pdb_file.write(pdb)
        pdb_io = StringIO(pdb)
        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure('structure', pdb_io)
        
        io = PDB.PDBIO()
        io.set_structure(structure)
        io.save(pdb_filename)


def load_existing_pdbs(titles, directory):
    pdbs = []
    for title in titles:
        pdb_filename = os.path.join(directory, f'{title}.pdb')
        if os.path.exists(pdb_filename):
            with open(pdb_filename, 'r') as pdb_file:
                pdbs.append(pdb_file.read())
        else:
            pdbs.append(None)
    return pdbs


def create_pdb_structure(protein_data):
    structure_id = protein_data['name']
    sequence = protein_data['seq']
    coords = protein_data['coords']
    
    structure = Structure.Structure(structure_id)
    model = Model.Model(0)
    chain = Chain.Chain('A')
    
    aa_map = {'A': 'ALA', 'C': 'CYS', 'D': 'ASP', 'E': 'GLU', 'F': 'PHE', 'G': 'GLY', 'H': 'HIS',
              'I': 'ILE', 'K': 'LYS', 'L': 'LEU', 'M': 'MET', 'N': 'ASN', 'P': 'PRO', 'Q': 'GLN', 
              'R': 'ARG', 'S': 'SER', 'T': 'THR', 'V': 'VAL', 'W': 'TRP', 'Y': 'TYR'}
    
    atom_names = ['N', 'CA', 'C', 'O']
    
    for res_index, (res, coord_set) in enumerate(zip(sequence, coords), start=1):
        residue = Residue.Residue((' ', res_index, ' '), aa_map[res], ' ')
        for atom_index, (atom_name, coord) in enumerate(zip(atom_names, coord_set)):
            atom = Atom(atom_name, coord, 1.0, 0.0, ' ', atom_name, atom_index, atom_name[0])
            residue.add(atom)
        chain.add(residue)
    
    model.add(chain)
    structure.add(model)
    return structure


class MInterface(MInterface_base):
    def __init__(self, model_name=None, loss=None, lr=None, **kargs):
        super().__init__()
        self.save_hyperparameters()
        self.load_model()
        self.cross_entropy = nn.NLLLoss(reduction='none')
        os.makedirs(os.path.join(self.hparams.res_dir, self.hparams.ex_name), exist_ok=True)

        self.test_step_outputs = []

        self.titles = []

        self.test_setupped = False

        self.esmfold_model = None

        self.inference_times = []

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


    def test_forward(self, batch):
        # Forward pass for test and additional metric calculations
        batch = self.model._get_features(batch)
        start_time = time.time()
        results = self.model(batch)
        end_time = time.time()
        self.inference_times.append(end_time - start_time)
        log_probs, mask = results['log_probs'], batch['mask']
        batch_ids = batch['batch_id']

        device = log_probs.device

        # Convert mask to boolean for indexing
        mask = mask > 0  # Convert mask to boolean (True where mask > 0)
        
        # Initialize lists to hold metrics for each sample
        losses = []
        recoveries = []
        plddt_ca_list = []
        plddt_list = []
        tmscores = []
        nssr_scores = []  # To store NSSR scores for each sample

        # Define directory to save the PDBs
        pdb_save_directory = f"predicted_pdb/{self.hparams.ex_name}/{self.hparams.dataset}"
        gt_pdb_save_directory = f"gt_pdb/{self.hparams.dataset}"

        # Ensure the ground truth PDB directory exists
        if not os.path.exists(gt_pdb_save_directory):
            os.makedirs(gt_pdb_save_directory)

        # Get the unique batch IDs (corresponding to different samples in the batch)
        unique_batch_ids = torch.unique(batch_ids)
        # print(unique_batch_ids)

        # Loop over each sample in the batch
        for sample_id in unique_batch_ids:
            # Get indices for the current sample
            sample_mask = batch_ids == sample_id

            # Apply the mask to get log_probs and ground truth S for this sample
            log_probs_sample = log_probs[sample_mask]
            mask_sample = mask[sample_mask]
            S_sample = batch['S'][sample_mask]

            # Further mask log_probs and S using the internal mask
            log_probs_masked = log_probs_sample[mask_sample]
            S_masked = S_sample[mask_sample]

            S_masked_list = S_masked.tolist()
            gt_tokens = tokenizer.convert_ids_to_tokens(S_masked_list, skip_special_tokens=True)  # Convert token IDs to tokens (amino acids or special tokens)
            # Join tokens into a single string and remove any special tokens if needed
            gt_amino_acid_sequence = "".join(gt_tokens)

            # Loss calculation (use masked log_probs and S)
            loss = self.cross_entropy(log_probs_masked, S_masked)
            losses.append(loss.mean())  # Append the loss for this sample

            # Recovery calculation (using the mask)
            predicted_indices = log_probs_masked.argmax(dim=-1)
            cmp = predicted_indices == S_masked
            recovery = cmp.float().mean()
            recoveries.append(recovery)  # Append the recovery metric

            predicted_indices_list = predicted_indices.tolist()
            pred_tokens = tokenizer.convert_ids_to_tokens(predicted_indices_list, skip_special_tokens=True)  # Convert token IDs to tokens (amino acids or special tokens)
            
            # Join tokens into a single string and remove any special tokens if needed
            pred_amino_acid_sequence = "".join(pred_tokens)

            # nssr
            # BLOSUM62-based NSSR calculation
            similar_pairs_count = 0
            total_residues = len(gt_amino_acid_sequence)
            # print('length:', total_residues)

            # Loop through each pair of residues in the predicted and ground truth sequences
            for gt_residue, pred_residue in zip(gt_amino_acid_sequence, pred_amino_acid_sequence):
                # Check if BLOSUM62 score is greater than 0
                if blosum62[gt_residue][pred_residue] > 0:
                    similar_pairs_count += 1

            # Calculate NSSR for the current sample
            if total_residues > 0:
                nssr_score = torch.tensor(similar_pairs_count / total_residues, device=device)
                nssr_scores.append(nssr_score)

            # Update perplexity for this sample
            self.perplexity_metric.update(log_probs_masked.unsqueeze(1), S_masked.unsqueeze(1))

            # Check if the PDB files already exist for the current sample
            sample_title = batch['title'][sample_id]
            existing_pdb = load_existing_pdbs([sample_title], pdb_save_directory)[0]
            pred_pdb_path = os.path.join(pdb_save_directory, f"{sample_title}.pdb")
            gt_pdb_path = os.path.join(gt_pdb_save_directory, f"{sample_title}.pdb")

            self.titles.append(sample_title)

            # Check if the ground truth PDB exists
            if not os.path.exists(gt_pdb_path):
                # Create the ground truth PDB from batch['X'] and amino_acid_sequence
                gt_coords = batch['X'][sample_mask][mask_sample].cpu().numpy()

                # Create the protein data structure
                protein_data = {
                    'name': sample_title,
                    'seq': gt_amino_acid_sequence,
                    'coords': gt_coords
                }
                # Create the structure using the function provided
                gt_structure = create_pdb_structure(protein_data)

                # Save the structure to the ground truth PDB file
                io = PDB.PDBIO()
                io.set_structure(gt_structure)
                io.save(gt_pdb_path)

            if existing_pdb is None:
                esmfold_inputs = esmfold_tokenizer([pred_amino_acid_sequence], return_tensors="pt", add_special_tokens=False)
                for k, v in esmfold_inputs.items():
                    esmfold_inputs[k] = v.to(device)
                esmfold_outputs = self.esmfold_model(**esmfold_inputs)

                # Convert outputs to PDB format and calculate pLDDT
                pdb = convert_outputs_to_pdb(esmfold_outputs)[0]
                # Save the newly generated PDB file
                save_pdbs_to_files([pdb], [sample_title], pdb_save_directory)
            else:
                pdb = existing_pdb

            tmp = calculate_average_plddt(pdb)
            plddt_ca = torch.tensor(tmp['plddt_ca'], device=device)
            plddt = torch.tensor(tmp['plddt'], device=device)

            plddt_ca_list.append(plddt_ca)
            plddt_list.append(plddt)

            # tm-score
            tmscore = torch.tensor(calculate_tm_score(pred_pdb_path, gt_pdb_path), device=device)
            tmscores.append(tmscore)

        return (
            losses, recoveries, plddt_ca_list, plddt_list, tmscores, nssr_scores
        )


    def on_test_epoch_end(self):
        # Calculate the average inference time per batch during testing
        if self.inference_times:
            avg_inference_time = statistics.mean(self.inference_times)
            std_inference_time = statistics.stdev(self.inference_times)
            print(f"Average inference time per batch: {avg_inference_time:.4f} seconds")
            print(f"Standard deviation of inference time per batch: {std_inference_time:.4f} seconds")

        model_device = next(self.model.parameters()).device
        # Compute average loss and recovery across all test batches
        avg_loss = torch.stack([x['test_loss'] for x in self.test_step_outputs]).mean().to(model_device)
        avg_recovery = torch.stack([x['test_recovery'] for x in self.test_step_outputs]).mean().to(model_device)
        avg_plddt_ca = torch.stack([x['test_plddt_ca'] for x in self.test_step_outputs]).mean().to(model_device)
        avg_plddt = torch.stack([x['test_plddt'] for x in self.test_step_outputs]).mean().to(model_device)
        avg_tmscore = torch.stack([x['test_tmscore'] for x in self.test_step_outputs]).mean().to(model_device)
        avg_nssr_score = torch.stack([x['test_nssr_score'] for x in self.test_step_outputs]).mean().to(model_device)

        # Compute perplexity over the entire test set
        perplexity = self.perplexity_metric.compute().to(model_device)

        # Log aggregated metrics for the entire test set
        self.log("test_loss", avg_loss, on_step=False, on_epoch=True, sync_dist=True, reduce_fx="mean")
        self.log("test_recovery", avg_recovery, on_step=False, on_epoch=True, sync_dist=True, reduce_fx="mean")
        self.log("test_perplexity", perplexity, on_step=False, on_epoch=True, sync_dist=True, )
        self.log("test_plddt_ca", avg_plddt_ca, on_step=False, on_epoch=True, sync_dist=True, reduce_fx="mean")
        self.log("test_plddt", avg_plddt, on_step=False, on_epoch=True, sync_dist=True, reduce_fx="mean")
        self.log("test_tmscore", avg_tmscore, on_step=False, on_epoch=True, sync_dist=True, reduce_fx="mean")
        self.log("test_nssr_score", avg_nssr_score, on_step=False, on_epoch=True, sync_dist=True, reduce_fx="mean")
        
        # save
        metrics_data = {
            'Recovery': [x['test_recovery'].cpu().numpy() for x in self.test_step_outputs],
            'NSSR': [x['test_nssr_score'].cpu().numpy() for x in self.test_step_outputs],
            'pLDDT_CA': [x['test_plddt_ca'].cpu().numpy() for x in self.test_step_outputs],
            'pLDDT': [x['test_plddt'].cpu().numpy() for x in self.test_step_outputs],
            'TMScore': [x['test_tmscore'].cpu().numpy() for x in self.test_step_outputs],
        }

        # Add partition data
        partitions_data = {
            'Title': self.titles,
        }
          
        # Combine metrics and partitions into one dictionary
        data = {**metrics_data, **partitions_data}
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Save to CSV file
        metrics_save_directory = f"test_results/{self.hparams.ex_name}/{self.hparams.dataset}"
        os.makedirs(metrics_save_directory, exist_ok=True)
        csv_save_path = os.path.join(metrics_save_directory, "test_results.csv")
        df.to_csv(csv_save_path, index=False)
        
        print(f"Metrics saved to {csv_save_path}")

        # Reset perplexity metric for future use (if necessary)
        self.perplexity_metric.reset()
        self.reset_metrics()

    def reset_metrics(self):
        self.test_step_outputs.clear()


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
        with torch.no_grad():
            losses, recoveries, plddt_cas, plddts, tmscores, nssr_scores = self.test_forward(batch)

            for i, loss in enumerate(losses):
                recovery = recoveries[i]
                plddt_ca = plddt_cas[i]
                plddt = plddts[i]
                tmscore = tmscores[i]
                nssr_score = nssr_scores[i]
                # surface_recovery = surface_recoveries[i]
                # core_recovery = core_recoveries[i]
                self.test_step_outputs.append({
                        "test_loss": loss,
                        "test_recovery": recovery,
                        'test_plddt_ca': plddt_ca, 
                        'test_plddt': plddt,
                        'test_tmscore': tmscore,
                        'test_nssr_score': nssr_score,
                    }
                )

        return 


    def test_setup(self):
        model_device = next(self.model.parameters()).device

        self.esmfold_model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", cache_dir="cache/transformers/esmforproteinfolding")
        self.esmfold_model.trunk.set_chunk_size(128)
        self.esmfold_model = self.esmfold_model.to(model_device)

        self.perplexity_metric = Perplexity(device=model_device)

        
    def load_model(self):
        params = OmegaConf.load(f'./src/models/configs/{self.hparams.model_name}.yaml')
        params.update(self.hparams)

        if self.hparams.model_name == 'SBC2Model':
            from src.models.SBC2_model import SBC2Model
            self.model = SBC2Model(params)

        # self.model_device = next(self.model.parameters()).device

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
