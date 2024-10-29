import sys; sys.path.append('/huyuqi/xmyu/DiffSDS')
import inspect
import torch
from src.tools.utils import cuda
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import os
from torcheval.metrics.text import Perplexity
from torcheval.metrics import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score, BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score
from src.interface.model_interface import MInterface_base
from src.models.MemoryPiFold import MemoPiFold_model
import math
import torch.nn.functional as F
from torch.cuda.amp import autocast
from omegaconf import OmegaConf
from transformers import AutoTokenizer, EsmForProteinFolding
from transformers.models.esm.openfold_utils.protein import to_pdb, Protein as OFProtein
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37
from Bio import PDB
from Bio.PDB import Superimposer, Structure, Model, Chain, Residue
from Bio.PDB.Atom import Atom
from Bio.Align import substitution_matrices
from Bio.PDB.SASA import ShrakeRupley
from io import StringIO
import subprocess
import requests
import time
import copy
import pandas as pd
import statistics

tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D", cache_dir="gaozhangyang/model_zoom/transformers") # mask token: 32
esmfold_tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1", cache_dir="cache/transformers/tokenizers")

# Load BLOSUM62 substitution matrix
blosum62 = substitution_matrices.load("BLOSUM62")

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

class_names = ['Alpha', 'Beta', 'AlphaBeta', 'FewSecondaryStructures', 'Other']  # CATH class names

bio_feat_dict = {
    "hydrophobicity": {
        "I": 4.5, "V": 4.2, "L": 3.8, "F": 2.8, "C": 2.5, "M": 1.9, "A": 1.8,
        "W": -0.9, "G": -0.4, "T": -0.7, "S": -0.8, "Y": -1.3, "P": -1.6, "H": -3.2,
        "N": -3.5, "D": -3.5, "Q": -3.5, "E": -3.5, "K": -3.9, "R": -4.5
    },
    "charge": {
        "R": 1, "K": 1, "D": -1, "E": -1, "H": 0.1, "A": 0, "C": 0, "F": 0, "G": 0, "I": 0,
        "L": 0, "M": 0, "N": 0, "P": 0, "Q": 0, "S": 0, "T": 0, "V": 0, "W": 0, "Y": 0
    },
    "polarity": {
        "R": 1, "N": 1, "D": 1, "Q": 1, "E": 1, "H": 1, "K": 1, "S": 1, "T": 1, "Y": 1,
        "A": 0, "C": 0, "F": 0, "G": 0, "I": 0, "L": 0, "M": 0, "P": 0, "V": 0, "W": 0
    },
    "acceptor": {
        "D": 1, "E": 1, "N": 1, "Q": 1, "H": 1, "S": 1, "T": 1, "Y": 1,
        "A": 0, "C": 0, "F": 0, "G": 0, "I": 0, "K": 0, "L": 0, "M": 0, "P": 0, "R": 0, "V": 0, "W": 0
    },
    "donor": {
        "R": 1, "K": 1, "W": 1, "N": 1, "Q": 1, "H": 1, "S": 1, "T": 1, "Y": 1,
        "A": 0, "C": 0, "D": 0, "E": 0, "F": 0, "G": 0, "I": 0, "L": 0, "M": 0, "P": 0, "V": 0
    }
}

three_to_one = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
    'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
    'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
    'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}


def fetch_cath_class(pdb_id):
    """ Fetch the CATH class of a protein by querying the CATH API """
    # Directory where CATH classes will be cached
    cath_class_dir = "cath_classes"
    os.makedirs(cath_class_dir, exist_ok=True)  # Create directory if it doesn't exist
    # Filepath to cache the CATH class
    cath_class_file = os.path.join(cath_class_dir, f"{pdb_id}.txt")
    
    # Check if the CATH class has already been cached
    if os.path.exists(cath_class_file):
        with open(cath_class_file, 'r') as f:
            cath_class = f.read().strip()  # Read and remove any trailing newlines
        return cath_class  # Return the cached CATH class
    
    max_domain_length = 0
    cath_class = 'Other'  # Default to 'Other'
    
    for i in range(10):
        pdb_id_variant = f"{pdb_id}{i:02d}"
        url = f"https://www.cathdb.info/version/v4_2_0/api/rest/domain_summary/{pdb_id_variant}"
        
        try:
            response = requests.get(url).json()
            if response['success'] and len(response['data']['residues']) > 0:
                domain_length = len(response['data']['residues'])
                # Get CATH class from the superfamily_id (split by '.' and take the 0-th element)
                domain_cath_class_id = response['data']['superfamily_id'].split('.')[0]

                # Map the CATH class ID to the corresponding CATH class name
                if domain_length > max_domain_length:
                    max_domain_length = domain_length
                    if domain_cath_class_id == '1':
                        cath_class = 'Alpha'
                    elif domain_cath_class_id == '2':
                        cath_class = 'Beta'
                    elif domain_cath_class_id == '3':
                        cath_class = 'AlphaBeta'
                    elif domain_cath_class_id == '4':
                        cath_class = 'FewSecondaryStructures'
                    else:
                        cath_class = 'Other'  # Non-standard classes
        except Exception as e:
            print(f"Error fetching CATH class for {pdb_id_variant}: {e}")


    # Cache the fetched CATH class to the file for future use
    with open(cath_class_file, 'w') as f:
        f.write(cath_class)
    
    return cath_class


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


def create_atoms_from_coords(coords):
    """
    Create mock Atom objects from given coordinates.
    Args:
    coords: numpy array of shape (N, 3) representing atomic coordinates.

    Returns:
    atoms: list of Atom objects.
    """
    atoms = []
    for i, coord in enumerate(coords):
        atom = Atom(name=str(i), coord=coord, bfactor=0.0, occupancy=1.0, altloc='', fullname=' CA ', serial_number=i+1, element='C')
        atoms.append(atom)
    return atoms


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


def get_ca_atoms_from_struc(structure, atom_type="CA"):
    """
    Extracts atoms of a given type (default is CA atoms) from a structure.
    """
    atoms = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if atom_type in residue:
                    atoms.append(residue[atom_type])
    return atoms


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


def calculate_contact_order(gt_ca_coords, distance_threshold=8.0):
    n = gt_ca_coords.size(0)

    # Step 1: Calculate the distance matrix
    distance_matrix = torch.norm(gt_ca_coords[:, None, :] - gt_ca_coords[None, :, :], dim=2)

    # Step 2: Determine contacts
    contacts = distance_matrix < distance_threshold

    # Step 3: Use the upper triangle of the contacts
    upper_triangle_contacts = contacts.triu(diagonal=1)

    # Calculate total distance and count
    contact_indices = upper_triangle_contacts.nonzero(as_tuple=True)
    total_distance = (contact_indices[1] - contact_indices[0]).abs().sum()
    count = contact_indices[0].numel()

    # Contact order calculation
    if count > 0:
        contact_order = total_distance / count / n
    else:
        contact_order = 0  # No contacts found

    return contact_order


class MInterface(MInterface_base):
    def __init__(self, model_name=None, loss=None, lr=None, **kargs):
        super().__init__()
        self.save_hyperparameters()
        self.load_model()
        self.cross_entropy = nn.NLLLoss(reduction='none')
        os.makedirs(os.path.join(self.hparams.res_dir, self.hparams.ex_name), exist_ok=True)

        self.test_step_outputs = []

        self.cath_classes = []  # CATH class partition
        self.sequence_lengths = []  # Sequence length partition (≤100, 100–300, >300)
        self.contact_orders = []  # Contact order for each protein
        self.titles = []

        self.surface_sizes = []
        self.core_sizes = []

        self.surface_recoveries_nan_included = []
        self.core_recoveries_nan_included = []

        self.surface_recoveries = []
        self.core_recoveries = []
        self.surface_nssrs = []
        self.core_nssrs = []

        self.tmscores_cath42_82 = []
        self.plddt_ca_cath42_82 = []
        self.plddt_cath42_82 = []

        # Metrics for sequences with length ≤ 100
        self.recovery_len_100 = []
        self.plddt_ca_len_100 = []
        self.plddt_len_100 = []
        self.rmsd_len_100 = []
        self.tmscore_len_100 = []
        self.nssr_len_100 = []
        self.hydrophobicity_rmsd_len_100 = []  # Hydrophobicity RMSD for length ≤ 100
        self.charge_rmsd_len_100 = []          # Charge RMSD for length ≤ 100

        # Metrics for sequences with length 100–300
        self.recovery_len_100_300 = []
        self.plddt_ca_len_100_300 = []
        self.plddt_len_100_300 = []
        self.rmsd_len_100_300 = []
        self.tmscore_len_100_300 = []
        self.nssr_len_100_300 = []
        self.hydrophobicity_rmsd_len_100_300 = []  # Hydrophobicity RMSD for length 100–300
        self.charge_rmsd_len_100_300 = []          # Charge RMSD for length 100–300

        # Metrics for sequences with length > 300
        self.recovery_len_300 = []
        self.plddt_ca_len_300 = []
        self.plddt_len_300 = []
        self.rmsd_len_300 = []
        self.tmscore_len_300 = []
        self.nssr_len_300 = []
        self.hydrophobicity_rmsd_len_300 = []  # Hydrophobicity RMSD for length > 300
        self.charge_rmsd_len_300 = []          # Charge RMSD for length > 300

        # Initialize lists to store metrics for each CATH class
        # Initialize metrics for CATH class Alpha
        self.recovery_alpha = []
        self.plddt_ca_alpha = []
        self.plddt_alpha = []
        self.rmsd_alpha = []
        self.tmscore_alpha = []
        self.nssr_alpha = []
        self.hydrophobicity_rmsd_alpha = []  # New
        self.charge_rmsd_alpha = []          # New

        # Initialize metrics for CATH class Beta
        self.recovery_beta = []
        self.plddt_ca_beta = []
        self.plddt_beta = []
        self.rmsd_beta = []
        self.tmscore_beta = []
        self.nssr_beta = []
        self.hydrophobicity_rmsd_beta = []  # New
        self.charge_rmsd_beta = []          # New

        # Initialize metrics for CATH class AlphaBeta
        self.recovery_alpha_beta = []
        self.plddt_ca_alpha_beta = []
        self.plddt_alpha_beta = []
        self.rmsd_alpha_beta = []
        self.tmscore_alpha_beta = []
        self.nssr_alpha_beta = []
        self.hydrophobicity_rmsd_alpha_beta = []  # New
        self.charge_rmsd_alpha_beta = []          # New

        # Initialize metrics for CATH class FewSecondaryStructures
        self.recovery_few_secondary_structures = []
        self.plddt_ca_few_secondary_structures = []
        self.plddt_few_secondary_structures = []
        self.rmsd_few_secondary_structures = []
        self.tmscore_few_secondary_structures = []
        self.nssr_few_secondary_structures = []
        self.hydrophobicity_rmsd_few_secondary_structures = []  # New
        self.charge_rmsd_few_secondary_structures = []          # New

        # Initialize metrics for CATH class Other
        self.recovery_other = []
        self.plddt_ca_other = []
        self.plddt_other = []
        self.rmsd_other = []
        self.tmscore_other = []
        self.nssr_other = []
        self.hydrophobicity_rmsd_other = []  # New
        self.charge_rmsd_other = []          # New

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
            if self.hparams.model_name == 'GVP':
                loss = self.cross_entropy(log_probs, batch.seq)
            else:
                loss = self.cross_entropy(log_probs, batch['S'])
            
            if self.hparams.model_name == 'AlphaDesign':
                loss += self.cross_entropy(results['log_probs0'], batch['S'])
            loss = (loss*mask).sum()/(mask.sum())

        if self.hparams.model_name == 'SBCModel':
            contrastive_loss = results['contrastive_loss']
            loss += contrastive_loss
            # loss = 0.5 * loss + 0.5 * contrastive_loss
        if self.hparams.model_name == 'SBC2Model':
            contrastive_loss = results['contrastive_loss']
            loss += contrastive_loss
            # loss = 2 * loss + contrastive_loss
        
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
        rmsds = []
        tmscores = []
        nssr_scores = []  # To store NSSR scores for each sample

        # Initialize lists to store biochemical RMSDs
        hydrophobicity_rmsds = []
        charge_rmsds = []

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

            # biochem rmsds
            # Initialize lists to hold hydrophobicity and charge values for RMSD calculations
            gt_hydrophobicity = []
            pred_hydrophobicity = []
            gt_charge = []
            pred_charge = []

            # Loop through each pair of residues in the predicted and ground truth sequences
            for gt_residue, pred_residue in zip(gt_amino_acid_sequence, pred_amino_acid_sequence):
                # Extract hydrophobicity and charge values from bio_feat_dict
                if gt_residue in bio_feat_dict['hydrophobicity'] and pred_residue in bio_feat_dict['hydrophobicity']:
                    gt_hydrophobicity.append(bio_feat_dict['hydrophobicity'][gt_residue])
                    pred_hydrophobicity.append(bio_feat_dict['hydrophobicity'][pred_residue])

                if gt_residue in bio_feat_dict['charge'] and pred_residue in bio_feat_dict['charge']:
                    gt_charge.append(bio_feat_dict['charge'][gt_residue])
                    pred_charge.append(bio_feat_dict['charge'][pred_residue])

            # Calculate hydrophobicity RMSD
            if len(gt_hydrophobicity) > 0:
                hydrophobicity_diff = torch.tensor(gt_hydrophobicity) - torch.tensor(pred_hydrophobicity)
                hydrophobicity_rmsd = torch.sqrt(torch.mean(hydrophobicity_diff ** 2))
                hydrophobicity_rmsds.append(hydrophobicity_rmsd)
            else:
                hydrophobicity_rmsd = None

            # Calculate charge RMSD
            if len(gt_charge) > 0:
                charge_diff = torch.tensor(gt_charge) - torch.tensor(pred_charge)
                charge_rmsd = torch.sqrt(torch.mean(charge_diff.float() ** 2))
                charge_rmsds.append(charge_rmsd)
            else:
                charge_rmsd = None

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

            # rmsd
            # Use StringIO to convert the PDB content (string) into a file-like object
            pdb_io = StringIO(pdb)
            parser = PDB.PDBParser(QUIET=True)
    
            # Parse the structure from the file-like object
            pred_structure = parser.get_structure('structure', pdb_io)
            pred_ca_atoms = get_ca_atoms_from_struc(pred_structure)

            gt_ca_coords = batch['X'][sample_mask][mask_sample][:, 1, :].cpu().numpy()
            gt_ca_atoms = create_atoms_from_coords(gt_ca_coords)

            # Perform the alignment
            super_imposer = Superimposer()
            super_imposer.set_atoms(pred_ca_atoms, gt_ca_atoms)

            # Calculate RMSD
            rmsd = torch.tensor(super_imposer.rms, device=device)
            rmsds.append(rmsd)

            # tm-score
            tmscore = torch.tensor(calculate_tm_score(pred_pdb_path, gt_pdb_path), device=device)
            tmscores.append(tmscore)

            # tm-score on cath 4.2 82
            cath_42_82_df = pd.read_excel('./cath_test_82/CATH42_82.xlsx')
            cath_42_82_name_list = cath_42_82_df['name'].tolist()
            # Add a '.' to the 4th position of each string
            cath_42_82_name_list = [name[:4] + '.' + name[4:] for name in cath_42_82_name_list]
            if sample_title in cath_42_82_name_list:
                self.tmscores_cath42_82.append(tmscore)
                self.plddt_ca_cath42_82.append(plddt_ca)
                self.plddt_cath42_82.append(plddt)

            # recovery for surface/core region
            # 1. 解析 PDB 文件
            parser = PDB.PDBParser()
            structure = parser.get_structure("protein", gt_pdb_path)

            # 2. 使用 Shrake-Rupley 算法计算 SASA
            sr = ShrakeRupley()
            sr.compute(structure, level="R")  # "R" 表示计算每个残基的 SASA

            # 3. 获取和分类每个氨基酸的 SASA 值
            surface_residues = []
            core_residues = []

            threshold = 30.0  # SASA 阈值 (单位：Å²)
            for model in structure:
                for chain in model:
                    for i, residue in enumerate(chain):
                        sasa = residue.sasa  # 每个残基的 SASA 值
                        # residue_letter = residue.get_resname()  # 获取三字母名称
                        # single_letter = three_to_one[residue_letter]  # 映射到单字母
                        
                        if sasa > threshold:
                            surface_residues.append(i)
                        else:
                            core_residues.append(i)

            surface_predicted_indices = predicted_indices[surface_residues]
            core_predicted_indices = predicted_indices[core_residues]
            surface_S_masked = S_masked[surface_residues]
            core_S_masked = S_masked[core_residues]
            cmp = surface_predicted_indices == surface_S_masked
            surface_recovery = cmp.float().mean()
            if not torch.isnan(surface_recovery):
                self.surface_recoveries.append(surface_recovery)
            self.surface_recoveries_nan_included.append(surface_recovery)
            self.surface_sizes.append(len(surface_predicted_indices))

            cmp = core_predicted_indices == core_S_masked
            core_recovery = cmp.float().mean()
            if not torch.isnan(core_recovery):
                self.core_recoveries.append(core_recovery)
            self.core_recoveries_nan_included.append(core_recovery)
            self.core_sizes.append(len(core_predicted_indices))

            # nssr for surface/core region
            surface_similar_pairs_count = 0
            # Loop through each pair of residues in the predicted and ground truth sequences
            for gt_residue, pred_residue in zip([gt_amino_acid_sequence[r] for r in surface_residues], [pred_amino_acid_sequence[r] for r in surface_residues]):
                # Check if BLOSUM62 score is greater than 0
                if blosum62[gt_residue][pred_residue] > 0:
                    surface_similar_pairs_count += 1
            # Calculate NSSR for the current sample
            if len(surface_residues) > 0:
                nssr_score = torch.tensor(surface_similar_pairs_count / len(surface_residues), device=device)
                self.surface_nssrs.append(nssr_score)

            core_similar_pairs_count = 0
            # Loop through each pair of residues in the predicted and ground truth sequences
            for gt_residue, pred_residue in zip([gt_amino_acid_sequence[r] for r in core_residues], [pred_amino_acid_sequence[r] for r in core_residues]):
                # Check if BLOSUM62 score is greater than 0
                if blosum62[gt_residue][pred_residue] > 0:
                    core_similar_pairs_count += 1
            # Calculate NSSR for the current sample
            if len(core_residues) > 0:
                nssr_score = torch.tensor(core_similar_pairs_count / len(core_residues), device=device)
                self.core_nssrs.append(nssr_score)

            # contact order
            contact_order = calculate_contact_order(batch['X'][sample_mask][mask_sample][:, 1, :])
            self.contact_orders.append(contact_order)

            # metrics for different length
            # Get the length of the ground truth sequence
            seq_length = len(gt_tokens)
            self.sequence_lengths.append(seq_length)

            # Store metrics in corresponding length categories
            if seq_length <= 100:
                self.recovery_len_100.append(recovery)
                self.plddt_ca_len_100.append(plddt_ca)
                self.plddt_len_100.append(plddt)
                self.rmsd_len_100.append(rmsd)
                self.tmscore_len_100.append(tmscore)
                self.nssr_len_100.append(nssr_score)
                if hydrophobicity_rmsd is not None:
                    self.hydrophobicity_rmsd_len_100.append(hydrophobicity_rmsd)
                if charge_rmsd is not None:
                    self.charge_rmsd_len_100.append(charge_rmsd)
            elif 100 < seq_length <= 300:
                self.recovery_len_100_300.append(recovery)
                self.plddt_ca_len_100_300.append(plddt_ca)
                self.plddt_len_100_300.append(plddt)
                self.rmsd_len_100_300.append(rmsd)
                self.tmscore_len_100_300.append(tmscore)
                self.nssr_len_100_300.append(nssr_score)
                if hydrophobicity_rmsd is not None:
                    self.hydrophobicity_rmsd_len_100_300.append(hydrophobicity_rmsd)
                if charge_rmsd is not None:
                    self.charge_rmsd_len_100_300.append(charge_rmsd)
            else:  # Length > 300
                self.recovery_len_300.append(recovery)
                self.plddt_ca_len_300.append(plddt_ca)
                self.plddt_len_300.append(plddt)
                self.rmsd_len_300.append(rmsd)
                self.tmscore_len_300.append(tmscore)
                self.nssr_len_300.append(nssr_score)
                if hydrophobicity_rmsd is not None:
                    self.hydrophobicity_rmsd_len_300.append(hydrophobicity_rmsd)
                if charge_rmsd is not None:
                    self.charge_rmsd_len_300.append(charge_rmsd)

            # residue-level metrics
            gt_tokens_indices = torch.tensor([residue_to_index[residue] for residue in gt_tokens], device=device)
            pred_tokens_indices = torch.tensor([residue_to_index[residue] for residue in pred_tokens], device=device)

            # Multiclass classification metrics (all residues)
            self.all_residue_accuracy.update(pred_tokens_indices, gt_tokens_indices)
            self.all_residue_precision.update(pred_tokens_indices, gt_tokens_indices)
            self.all_residue_recall.update(pred_tokens_indices, gt_tokens_indices)
            self.all_residue_f1.update(pred_tokens_indices, gt_tokens_indices)

            # Convert the ground truth and predicted tokens to NumPy arrays
            gt_amino_acid_sequence = np.array(gt_tokens)  # Ground truth sequence as a NumPy array
            pred_amino_acid_sequence = np.array(pred_tokens)  # Predicted sequence as a NumPy array

            # Binary classification for each residue type
            for residue_type in residue_types:
                # Create binary arrays using vectorized comparison
                gt_binary = torch.tensor((gt_amino_acid_sequence == residue_type).astype(int), device=device)
                pred_binary = torch.tensor((pred_amino_acid_sequence == residue_type).astype(int), device=device)

                # Update binary accuracy, precision, recall, F1 metrics for this residue type
                self.binary_accuracies[residue_type].update(pred_binary, gt_binary)
                self.binary_precisions[residue_type].update(pred_binary, gt_binary)
                self.binary_recalls[residue_type].update(pred_binary, gt_binary)
                self.binary_f1s[residue_type].update(pred_binary, gt_binary)

            if self.hparams.dataset == 'CATH4.2SurfProPiFoldDense':
                # different cath classes
                # Get the CATH class for the protein based on the PDB ID
                pdb_id = sample_title.replace('.', '')[:5]  # Remove '.' and get first 5 chars
                cath_class = fetch_cath_class(pdb_id)
                self.cath_classes.append(cath_class)

                # Append the metrics to the appropriate list based on the CATH class
                # Append the metrics to the appropriate list based on the CATH class
                if cath_class == 'Alpha':
                    self.recovery_alpha.append(recovery)
                    self.plddt_ca_alpha.append(plddt_ca)
                    self.plddt_alpha.append(plddt)
                    self.rmsd_alpha.append(rmsd)
                    self.tmscore_alpha.append(tmscore)
                    self.nssr_alpha.append(nssr_score)
                    if hydrophobicity_rmsd is not None:  # Check if hydrophobicity_rmsd is valid
                        self.hydrophobicity_rmsd_alpha.append(hydrophobicity_rmsd)
                    if charge_rmsd is not None:  # Check if charge_rmsd is valid
                        self.charge_rmsd_alpha.append(charge_rmsd)

                elif cath_class == 'Beta':
                    self.recovery_beta.append(recovery)
                    self.plddt_ca_beta.append(plddt_ca)
                    self.plddt_beta.append(plddt)
                    self.rmsd_beta.append(rmsd)
                    self.tmscore_beta.append(tmscore)
                    self.nssr_beta.append(nssr_score)
                    if hydrophobicity_rmsd is not None:
                        self.hydrophobicity_rmsd_beta.append(hydrophobicity_rmsd)
                    if charge_rmsd is not None:
                        self.charge_rmsd_beta.append(charge_rmsd)

                elif cath_class == 'AlphaBeta':
                    self.recovery_alpha_beta.append(recovery)
                    self.plddt_ca_alpha_beta.append(plddt_ca)
                    self.plddt_alpha_beta.append(plddt)
                    self.rmsd_alpha_beta.append(rmsd)
                    self.tmscore_alpha_beta.append(tmscore)
                    self.nssr_alpha_beta.append(nssr_score)
                    if hydrophobicity_rmsd is not None:
                        self.hydrophobicity_rmsd_alpha_beta.append(hydrophobicity_rmsd)
                    if charge_rmsd is not None:
                        self.charge_rmsd_alpha_beta.append(charge_rmsd)

                elif cath_class == 'FewSecondaryStructures':
                    self.recovery_few_secondary_structures.append(recovery)
                    self.plddt_ca_few_secondary_structures.append(plddt_ca)
                    self.plddt_few_secondary_structures.append(plddt)
                    self.rmsd_few_secondary_structures.append(rmsd)
                    self.tmscore_few_secondary_structures.append(tmscore)
                    self.nssr_few_secondary_structures.append(nssr_score)
                    if hydrophobicity_rmsd is not None:
                        self.hydrophobicity_rmsd_few_secondary_structures.append(hydrophobicity_rmsd)
                    if charge_rmsd is not None:
                        self.charge_rmsd_few_secondary_structures.append(charge_rmsd)

                else:  # 'Other' class
                    self.recovery_other.append(recovery)
                    self.plddt_ca_other.append(plddt_ca)
                    self.plddt_other.append(plddt)
                    self.rmsd_other.append(rmsd)
                    self.tmscore_other.append(tmscore)
                    self.nssr_other.append(nssr_score)
                    if hydrophobicity_rmsd is not None:
                        self.hydrophobicity_rmsd_other.append(hydrophobicity_rmsd)
                    if charge_rmsd is not None:
                        self.charge_rmsd_other.append(charge_rmsd)

        return (
            losses, recoveries, plddt_ca_list, plddt_list, rmsds, tmscores, nssr_scores, hydrophobicity_rmsds, 
            charge_rmsds, 
            # surface_recoveries, core_recoveries,
        )


    def on_test_epoch_end(self):
        # Calculate the average inference time per batch during testing
        if self.inference_times:
            avg_inference_time = statistics.mean(self.inference_times)
            std_inference_time = statistics.stdev(self.inference_times)
            print(f"Average inference time per batch: {avg_inference_time:.4f} seconds")
            print(f"Standard deviation of inference time per batch: {std_inference_time:.4f} seconds")

        def compute_avg(metric_list):
            return torch.stack(metric_list).mean().to(model_device) if metric_list else torch.tensor(0.0).to(model_device)

        model_device = next(self.model.parameters()).device
        # Compute average loss and recovery across all test batches
        avg_loss = torch.stack([x['test_loss'] for x in self.test_step_outputs]).mean().to(model_device)
        avg_recovery = torch.stack([x['test_recovery'] for x in self.test_step_outputs]).mean().to(model_device)
        avg_plddt_ca = torch.stack([x['test_plddt_ca'] for x in self.test_step_outputs]).mean().to(model_device)
        avg_plddt = torch.stack([x['test_plddt'] for x in self.test_step_outputs]).mean().to(model_device)
        avg_rmsd = torch.stack([x['test_rmsd'] for x in self.test_step_outputs]).mean().to(model_device)
        avg_tmscore = torch.stack([x['test_tmscore'] for x in self.test_step_outputs]).mean().to(model_device)
        avg_nssr_score = torch.stack([x['test_nssr_score'] for x in self.test_step_outputs]).mean().to(model_device)
        avg_hydrophobicity_rmsd = torch.stack([x['test_hydrophobicity_rmsd'] for x in self.test_step_outputs]).mean().to(model_device)
        avg_charge_rmsd = torch.stack([x['test_charge_rmsd'] for x in self.test_step_outputs]).mean().to(model_device)
        # avg_surface_recovery = torch.stack([x['test_surface_recovery'] for x in self.test_step_outputs]).mean().to(model_device)
        # avg_core_recovery = torch.stack([x['test_core_recovery'] for x in self.test_step_outputs]).mean().to(model_device)

        avg_surface_recovery = compute_avg(self.surface_recoveries)
        avg_core_recovery = compute_avg(self.core_recoveries)

        # Compute perplexity over the entire test set
        perplexity = self.perplexity_metric.compute().to(model_device)

        # Log aggregated metrics for the entire test set
        self.log("test_loss", avg_loss, on_step=False, on_epoch=True, sync_dist=True, reduce_fx="mean")
        self.log("test_recovery", avg_recovery, on_step=False, on_epoch=True, sync_dist=True, reduce_fx="mean")
        self.log("test_perplexity", perplexity, on_step=False, on_epoch=True, sync_dist=True, )
        self.log("test_plddt_ca", avg_plddt_ca, on_step=False, on_epoch=True, sync_dist=True, reduce_fx="mean")
        self.log("test_plddt", avg_plddt, on_step=False, on_epoch=True, sync_dist=True, reduce_fx="mean")
        self.log("test_rmsd", avg_rmsd, on_step=False, on_epoch=True, sync_dist=True, reduce_fx="mean")
        self.log("test_tmscore", avg_tmscore, on_step=False, on_epoch=True, sync_dist=True, reduce_fx="mean")
        self.log("test_nssr_score", avg_nssr_score, on_step=False, on_epoch=True, sync_dist=True, reduce_fx="mean")
        self.log("test_hydrophobicity_rmsd", avg_hydrophobicity_rmsd, on_step=False, on_epoch=True, sync_dist=True, reduce_fx="mean")
        self.log("test_charge_rmsd", avg_charge_rmsd, on_step=False, on_epoch=True, sync_dist=True, reduce_fx="mean")
        self.log("test_surface_recovery", avg_surface_recovery, on_step=False, on_epoch=True, sync_dist=True, reduce_fx="mean")
        self.log("test_core_recovery", avg_core_recovery, on_step=False, on_epoch=True, sync_dist=True, reduce_fx="mean")

        avg_tmscore_cath42_82 = compute_avg(self.tmscores_cath42_82)
        avg_plddt_ca_cath42_82 = compute_avg(self.plddt_ca_cath42_82)
        avg_plddt_cath42_82 = compute_avg(self.plddt_cath42_82)
        print('number of samples in cath_42_82: ', len(self.tmscores_cath42_82))
        self.log("test_tmscore_cath42_82", avg_tmscore_cath42_82, on_epoch=True, sync_dist=True)
        self.log("test_plddt_ca_cath42_82", avg_plddt_ca_cath42_82, on_epoch=True, sync_dist=True)
        self.log("test_plddt_cath42_82", avg_plddt_cath42_82, on_epoch=True, sync_dist=True)

        # Aggregate metrics for sequences with length ≤ 100
        avg_recovery_len_100 = compute_avg(self.recovery_len_100)
        avg_plddt_ca_len_100 = compute_avg(self.plddt_ca_len_100)
        avg_plddt_len_100 = compute_avg(self.plddt_len_100)
        avg_rmsd_len_100 = compute_avg(self.rmsd_len_100)
        avg_tmscore_len_100 = compute_avg(self.tmscore_len_100)
        avg_nssr_score_len_100 = compute_avg(self.nssr_len_100)
        avg_hydrophobicity_rmsd_len_100 = compute_avg(self.hydrophobicity_rmsd_len_100)  # New
        avg_charge_rmsd_len_100 = compute_avg(self.charge_rmsd_len_100)  # New

        # Aggregate metrics for sequences with length 100–300
        avg_recovery_len_100_300 = compute_avg(self.recovery_len_100_300)
        avg_plddt_ca_len_100_300 = compute_avg(self.plddt_ca_len_100_300)
        avg_plddt_len_100_300 = compute_avg(self.plddt_len_100_300)
        avg_rmsd_len_100_300 = compute_avg(self.rmsd_len_100_300)
        avg_tmscore_len_100_300 = compute_avg(self.tmscore_len_100_300)
        avg_nssr_score_len_100_300 = compute_avg(self.nssr_len_100_300)
        avg_hydrophobicity_rmsd_len_100_300 = compute_avg(self.hydrophobicity_rmsd_len_100_300)  # New
        avg_charge_rmsd_len_100_300 = compute_avg(self.charge_rmsd_len_100_300)  # New

        # Aggregate metrics for sequences with length > 300
        avg_recovery_len_300 = compute_avg(self.recovery_len_300)
        avg_plddt_ca_len_300 = compute_avg(self.plddt_ca_len_300)
        avg_plddt_len_300 = compute_avg(self.plddt_len_300)
        avg_rmsd_len_300 = compute_avg(self.rmsd_len_300)
        avg_tmscore_len_300 = compute_avg(self.tmscore_len_300)
        avg_nssr_score_len_300 = compute_avg(self.nssr_len_300)
        avg_hydrophobicity_rmsd_len_300 = compute_avg(self.hydrophobicity_rmsd_len_300)  # New
        avg_charge_rmsd_len_300 = compute_avg(self.charge_rmsd_len_300)  # New

        # Log aggregated metrics for sequences with length ≤ 100
        self.log("test_recovery_len_100", avg_recovery_len_100, on_epoch=True, sync_dist=True)
        self.log("test_plddt_ca_len_100", avg_plddt_ca_len_100, on_epoch=True, sync_dist=True)
        self.log("test_plddt_len_100", avg_plddt_len_100, on_epoch=True, sync_dist=True)
        self.log("test_rmsd_len_100", avg_rmsd_len_100, on_epoch=True, sync_dist=True)
        self.log("test_tmscore_len_100", avg_tmscore_len_100, on_epoch=True, sync_dist=True)
        self.log("test_nssr_score_len_100", avg_nssr_score_len_100, on_epoch=True, sync_dist=True)
        self.log("test_hydrophobicity_rmsd_len_100", avg_hydrophobicity_rmsd_len_100, on_epoch=True, sync_dist=True)  # New
        self.log("test_charge_rmsd_len_100", avg_charge_rmsd_len_100, on_epoch=True, sync_dist=True)  # New

        # Log aggregated metrics for sequences with length 100–300
        self.log("test_recovery_len_100_300", avg_recovery_len_100_300, on_epoch=True, sync_dist=True)
        self.log("test_plddt_ca_len_100_300", avg_plddt_ca_len_100_300, on_epoch=True, sync_dist=True)
        self.log("test_plddt_len_100_300", avg_plddt_len_100_300, on_epoch=True, sync_dist=True)
        self.log("test_rmsd_len_100_300", avg_rmsd_len_100_300, on_epoch=True, sync_dist=True)
        self.log("test_tmscore_len_100_300", avg_tmscore_len_100_300, on_epoch=True, sync_dist=True)
        self.log("test_nssr_score_len_100_300", avg_nssr_score_len_100_300, on_epoch=True, sync_dist=True)
        self.log("test_hydrophobicity_rmsd_len_100_300", avg_hydrophobicity_rmsd_len_100_300, on_epoch=True, sync_dist=True)  # New
        self.log("test_charge_rmsd_len_100_300", avg_charge_rmsd_len_100_300, on_epoch=True, sync_dist=True)  # New

        # Log aggregated metrics for sequences with length > 300
        self.log("test_recovery_len_300", avg_recovery_len_300, on_epoch=True, sync_dist=True)
        self.log("test_plddt_ca_len_300", avg_plddt_ca_len_300, on_epoch=True, sync_dist=True)
        self.log("test_plddt_len_300", avg_plddt_len_300, on_epoch=True, sync_dist=True)
        self.log("test_rmsd_len_300", avg_rmsd_len_300, on_epoch=True, sync_dist=True)
        self.log("test_tmscore_len_300", avg_tmscore_len_300, on_epoch=True, sync_dist=True)
        self.log("test_nssr_score_len_300", avg_nssr_score_len_300, on_epoch=True, sync_dist=True)
        self.log("test_hydrophobicity_rmsd_len_300", avg_hydrophobicity_rmsd_len_300, on_epoch=True, sync_dist=True)  # New
        self.log("test_charge_rmsd_len_300", avg_charge_rmsd_len_300, on_epoch=True, sync_dist=True)  # New

        # Calculate average metrics for each CATH class
        # Compute averages for CATH class Alpha
        avg_recovery_alpha = compute_avg(self.recovery_alpha)
        avg_plddt_ca_alpha = compute_avg(self.plddt_ca_alpha)
        avg_plddt_alpha = compute_avg(self.plddt_alpha)
        avg_rmsd_alpha = compute_avg(self.rmsd_alpha)
        avg_tmscore_alpha = compute_avg(self.tmscore_alpha)
        avg_nssr_alpha = compute_avg(self.nssr_alpha)
        avg_hydrophobicity_rmsd_alpha = compute_avg(self.hydrophobicity_rmsd_alpha)  # New
        avg_charge_rmsd_alpha = compute_avg(self.charge_rmsd_alpha)  # New

        # Compute averages for CATH class Beta
        avg_recovery_beta = compute_avg(self.recovery_beta)
        avg_plddt_ca_beta = compute_avg(self.plddt_ca_beta)
        avg_plddt_beta = compute_avg(self.plddt_beta)
        avg_rmsd_beta = compute_avg(self.rmsd_beta)
        avg_tmscore_beta = compute_avg(self.tmscore_beta)
        avg_nssr_beta = compute_avg(self.nssr_beta)
        avg_hydrophobicity_rmsd_beta = compute_avg(self.hydrophobicity_rmsd_beta)  # New
        avg_charge_rmsd_beta = compute_avg(self.charge_rmsd_beta)  # New

        # Compute averages for CATH class AlphaBeta
        avg_recovery_alpha_beta = compute_avg(self.recovery_alpha_beta)
        avg_plddt_ca_alpha_beta = compute_avg(self.plddt_ca_alpha_beta)
        avg_plddt_alpha_beta = compute_avg(self.plddt_alpha_beta)
        avg_rmsd_alpha_beta = compute_avg(self.rmsd_alpha_beta)
        avg_tmscore_alpha_beta = compute_avg(self.tmscore_alpha_beta)
        avg_nssr_alpha_beta = compute_avg(self.nssr_alpha_beta)
        avg_hydrophobicity_rmsd_alpha_beta = compute_avg(self.hydrophobicity_rmsd_alpha_beta)  # New
        avg_charge_rmsd_alpha_beta = compute_avg(self.charge_rmsd_alpha_beta)  # New

        # Compute averages for CATH class FewSecondaryStructures
        avg_recovery_few_secondary_structures = compute_avg(self.recovery_few_secondary_structures)
        avg_plddt_ca_few_secondary_structures = compute_avg(self.plddt_ca_few_secondary_structures)
        avg_plddt_few_secondary_structures = compute_avg(self.plddt_few_secondary_structures)
        avg_rmsd_few_secondary_structures = compute_avg(self.rmsd_few_secondary_structures)
        avg_tmscore_few_secondary_structures = compute_avg(self.tmscore_few_secondary_structures)
        avg_nssr_few_secondary_structures = compute_avg(self.nssr_few_secondary_structures)
        avg_hydrophobicity_rmsd_few_secondary_structures = compute_avg(self.hydrophobicity_rmsd_few_secondary_structures)  # New
        avg_charge_rmsd_few_secondary_structures = compute_avg(self.charge_rmsd_few_secondary_structures)  # New

        # Compute averages for CATH class Other
        avg_recovery_other = compute_avg(self.recovery_other)
        avg_plddt_ca_other = compute_avg(self.plddt_ca_other)
        avg_plddt_other = compute_avg(self.plddt_other)
        avg_rmsd_other = compute_avg(self.rmsd_other)
        avg_tmscore_other = compute_avg(self.tmscore_other)
        avg_nssr_other = compute_avg(self.nssr_other)
        avg_hydrophobicity_rmsd_other = compute_avg(self.hydrophobicity_rmsd_other)  # New
        avg_charge_rmsd_other = compute_avg(self.charge_rmsd_other)  # New

        # Log metrics for CATH class Alpha
        self.log("test_recovery_alpha", avg_recovery_alpha, on_epoch=True, sync_dist=True)
        self.log("test_plddt_ca_alpha", avg_plddt_ca_alpha, on_epoch=True, sync_dist=True)
        self.log("test_plddt_alpha", avg_plddt_alpha, on_epoch=True, sync_dist=True)
        self.log("test_rmsd_alpha", avg_rmsd_alpha, on_epoch=True, sync_dist=True)
        self.log("test_tmscore_alpha", avg_tmscore_alpha, on_epoch=True, sync_dist=True)
        self.log("test_nssr_alpha", avg_nssr_alpha, on_epoch=True, sync_dist=True)
        self.log("test_hydrophobicity_rmsd_alpha", avg_hydrophobicity_rmsd_alpha, on_epoch=True, sync_dist=True)  # New
        self.log("test_charge_rmsd_alpha", avg_charge_rmsd_alpha, on_epoch=True, sync_dist=True)  # New

        # Log metrics for CATH class Beta
        self.log("test_recovery_beta", avg_recovery_beta, on_epoch=True, sync_dist=True)
        self.log("test_plddt_ca_beta", avg_plddt_ca_beta, on_epoch=True, sync_dist=True)
        self.log("test_plddt_beta", avg_plddt_beta, on_epoch=True, sync_dist=True)
        self.log("test_rmsd_beta", avg_rmsd_beta, on_epoch=True, sync_dist=True)
        self.log("test_tmscore_beta", avg_tmscore_beta, on_epoch=True, sync_dist=True)
        self.log("test_nssr_beta", avg_nssr_beta, on_epoch=True, sync_dist=True)
        self.log("test_hydrophobicity_rmsd_beta", avg_hydrophobicity_rmsd_beta, on_epoch=True, sync_dist=True)  # New
        self.log("test_charge_rmsd_beta", avg_charge_rmsd_beta, on_epoch=True, sync_dist=True)  # New

        # Log metrics for CATH class AlphaBeta
        self.log("test_recovery_alpha_beta", avg_recovery_alpha_beta, on_epoch=True, sync_dist=True)
        self.log("test_plddt_ca_alpha_beta", avg_plddt_ca_alpha_beta, on_epoch=True, sync_dist=True)
        self.log("test_plddt_alpha_beta", avg_plddt_alpha_beta, on_epoch=True, sync_dist=True)
        self.log("test_rmsd_alpha_beta", avg_rmsd_alpha_beta, on_epoch=True, sync_dist=True)
        self.log("test_tmscore_alpha_beta", avg_tmscore_alpha_beta, on_epoch=True, sync_dist=True)
        self.log("test_nssr_alpha_beta", avg_nssr_alpha_beta, on_epoch=True, sync_dist=True)
        self.log("test_hydrophobicity_rmsd_alpha_beta", avg_hydrophobicity_rmsd_alpha_beta, on_epoch=True, sync_dist=True)  # New
        self.log("test_charge_rmsd_alpha_beta", avg_charge_rmsd_alpha_beta, on_epoch=True, sync_dist=True)  # New

        # Log metrics for CATH class FewSecondaryStructures
        self.log("test_recovery_few_secondary_structures", avg_recovery_few_secondary_structures, on_epoch=True, sync_dist=True)
        self.log("test_plddt_ca_few_secondary_structures", avg_plddt_ca_few_secondary_structures, on_epoch=True, sync_dist=True)
        self.log("test_plddt_few_secondary_structures", avg_plddt_few_secondary_structures, on_epoch=True, sync_dist=True)
        self.log("test_rmsd_few_secondary_structures", avg_rmsd_few_secondary_structures, on_epoch=True, sync_dist=True)
        self.log("test_tmscore_few_secondary_structures", avg_tmscore_few_secondary_structures, on_epoch=True, sync_dist=True)
        self.log("test_nssr_few_secondary_structures", avg_nssr_few_secondary_structures, on_epoch=True, sync_dist=True)
        self.log("test_hydrophobicity_rmsd_few_secondary_structures", avg_hydrophobicity_rmsd_few_secondary_structures, on_epoch=True, sync_dist=True)  # New
        self.log("test_charge_rmsd_few_secondary_structures", avg_charge_rmsd_few_secondary_structures, on_epoch=True, sync_dist=True)  # New

        # Log metrics for CATH class Other
        self.log("test_recovery_other", avg_recovery_other, on_epoch=True, sync_dist=True)
        self.log("test_plddt_ca_other", avg_plddt_ca_other, on_epoch=True, sync_dist=True)
        self.log("test_plddt_other", avg_plddt_other, on_epoch=True, sync_dist=True)
        self.log("test_rmsd_other", avg_rmsd_other, on_epoch=True, sync_dist=True)
        self.log("test_tmscore_other", avg_tmscore_other, on_epoch=True, sync_dist=True)
        self.log("test_nssr_other", avg_nssr_other, on_epoch=True, sync_dist=True)
        self.log("test_hydrophobicity_rmsd_other", avg_hydrophobicity_rmsd_other, on_epoch=True, sync_dist=True)  # New
        self.log("test_charge_rmsd_other", avg_charge_rmsd_other, on_epoch=True, sync_dist=True)  # New

        # Compute residue-level metrics (for all residues)
        all_residue_accuracy = self.all_residue_accuracy.compute().to(model_device)
        all_residue_precision = self.all_residue_precision.compute().to(model_device)
        all_residue_recall = self.all_residue_recall.compute().to(model_device)
        all_residue_f1 = self.all_residue_f1.compute().to(model_device)

        # Log residue-level metrics (overall)
        self.log("test_all_residue_accuracy", all_residue_accuracy, on_epoch=True, sync_dist=True)
        self.log("test_all_residue_precision", all_residue_precision, on_epoch=True, sync_dist=True)
        self.log("test_all_residue_recall", all_residue_recall, on_epoch=True, sync_dist=True)
        self.log("test_all_residue_f1", all_residue_f1, on_epoch=True, sync_dist=True)

        # Compute and log binary classification metrics for each residue type
        for residue_type in residue_types:
            binary_accuracy = self.binary_accuracies[residue_type].compute().to(model_device)
            binary_precision = self.binary_precisions[residue_type].compute().to(model_device)
            binary_recall = self.binary_recalls[residue_type].compute().to(model_device)
            binary_f1 = self.binary_f1s[residue_type].compute().to(model_device)

            # Log the binary classification metrics for this residue type
            self.log(f"test_{residue_type}_accuracy", binary_accuracy, on_epoch=True, sync_dist=True)
            self.log(f"test_{residue_type}_precision", binary_precision, on_epoch=True, sync_dist=True)
            self.log(f"test_{residue_type}_recall", binary_recall, on_epoch=True, sync_dist=True)
            self.log(f"test_{residue_type}_f1", binary_f1, on_epoch=True, sync_dist=True)

        # save
        metrics_data = {
            'Recovery': [x['test_recovery'].cpu().numpy() for x in self.test_step_outputs],
            'NSSR': [x['test_nssr_score'].cpu().numpy() for x in self.test_step_outputs],
            'pLDDT_CA': [x['test_plddt_ca'].cpu().numpy() for x in self.test_step_outputs],
            'pLDDT': [x['test_plddt'].cpu().numpy() for x in self.test_step_outputs],
            'RMSD': [x['test_rmsd'].cpu().numpy() for x in self.test_step_outputs],
            'TMScore': [x['test_tmscore'].cpu().numpy() for x in self.test_step_outputs],
            'Surface Recovery': torch.stack(self.surface_recoveries_nan_included).cpu().numpy(),
            'Core Recovery': torch.stack(self.core_recoveries_nan_included).cpu().numpy(),
        }
        if self.hparams.dataset == 'CATH4.2SurfProPiFoldDense':
            # Add partition data
            partitions_data = {
                'CATH_Class': self.cath_classes,
                'Sequence_Length': self.sequence_lengths,
                'Contact_Order': torch.stack(self.contact_orders).cpu().numpy(),
                'Core_size': self.core_sizes,
                'Surface_size': self.surface_sizes,
                'Title': self.titles,
            }
        else:
            partitions_data = {
                'Sequence_Length': self.sequence_lengths,
                'Contact_Order': torch.stack(self.contact_orders).cpu().numpy(),
                'Core_size': self.core_sizes,
                'Surface_size': self.surface_sizes,
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

        # surface / core
        # Convert and save each list to CSV
        file_names = [
            ("surface_recoveries.csv", self.surface_recoveries),
            ("core_recoveries.csv", self.core_recoveries),
            ("surface_nssrs.csv", self.surface_nssrs),
            ("core_nssrs.csv", self.core_nssrs)
        ]

        for file_name, tensor_list in file_names:
            np_array = torch.stack(tensor_list).cpu().numpy()
            np.savetxt(os.path.join(metrics_save_directory, file_name), np_array, delimiter=',')

        # Reset the metrics for the next test run
        self.all_residue_accuracy.reset()
        self.all_residue_precision.reset()
        self.all_residue_recall.reset()
        self.all_residue_f1.reset()

        for metric in [self.binary_precisions, self.binary_recalls, self.binary_f1s]:
            for m in metric.values():
                m.reset()

        # Reset perplexity metric for future use (if necessary)
        self.perplexity_metric.reset()
        self.reset_metrics()

    def reset_metrics(self):
        """Resets the lists used for length-based metric aggregation."""
        self.test_step_outputs.clear()

        # Clear metrics for sequences with length ≤ 100
        self.recovery_len_100.clear()
        self.plddt_ca_len_100.clear()
        self.plddt_len_100.clear()
        self.rmsd_len_100.clear()
        self.tmscore_len_100.clear()
        self.nssr_len_100.clear()
        self.hydrophobicity_rmsd_len_100.clear()  # New
        self.charge_rmsd_len_100.clear()          # New

        # Clear metrics for sequences with length 100–300
        self.recovery_len_100_300.clear()
        self.plddt_ca_len_100_300.clear()
        self.plddt_len_100_300.clear()
        self.rmsd_len_100_300.clear()
        self.tmscore_len_100_300.clear()
        self.nssr_len_100_300.clear()
        self.hydrophobicity_rmsd_len_100_300.clear()  # New
        self.charge_rmsd_len_100_300.clear()          # New

        # Clear metrics for sequences with length > 300
        self.recovery_len_300.clear()
        self.plddt_ca_len_300.clear()
        self.plddt_len_300.clear()
        self.rmsd_len_300.clear()
        self.tmscore_len_300.clear()
        self.nssr_len_300.clear()
        self.hydrophobicity_rmsd_len_300.clear()  # New
        self.charge_rmsd_len_300.clear()          # New

        # Reset CATH class metrics
        self.recovery_alpha.clear()
        self.plddt_ca_alpha.clear()
        self.plddt_alpha.clear()
        self.rmsd_alpha.clear()
        self.tmscore_alpha.clear()
        self.nssr_alpha.clear()

        self.recovery_beta.clear()
        self.plddt_ca_beta.clear()
        self.plddt_beta.clear()
        self.rmsd_beta.clear()
        self.tmscore_beta.clear()
        self.nssr_beta.clear()

        self.recovery_alpha_beta.clear()
        self.plddt_ca_alpha_beta.clear()
        self.plddt_alpha_beta.clear()
        self.rmsd_alpha_beta.clear()
        self.tmscore_alpha_beta.clear()
        self.nssr_alpha_beta.clear()

        self.recovery_few_secondary_structures.clear()
        self.plddt_ca_few_secondary_structures.clear()
        self.plddt_few_secondary_structures.clear()
        self.rmsd_few_secondary_structures.clear()
        self.tmscore_few_secondary_structures.clear()
        self.nssr_few_secondary_structures.clear()

        self.recovery_other.clear()
        self.plddt_ca_other.clear()
        self.plddt_other.clear()
        self.rmsd_other.clear()
        self.tmscore_other.clear()
        self.nssr_other.clear()


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


    def test_setup(self):
        model_device = next(self.model.parameters()).device

        self.esmfold_model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", cache_dir="cache/transformers/esmforproteinfolding")
        self.esmfold_model.trunk.set_chunk_size(128)
        self.esmfold_model = self.esmfold_model.to(model_device)

        self.perplexity_metric = Perplexity(device=model_device)

        # Initialize metrics
        self.all_residue_accuracy = MulticlassAccuracy(num_classes=len(residue_types), average="macro", device=model_device)
        self.all_residue_precision = MulticlassPrecision(num_classes=len(residue_types), average="macro", device=model_device)
        self.all_residue_recall = MulticlassRecall(num_classes=len(residue_types), average="macro", device=model_device)
        self.all_residue_f1 = MulticlassF1Score(num_classes=len(residue_types), average="macro", device=model_device)
        # Initialize binary classification metrics for each residue type
        self.binary_accuracies = {residue: BinaryAccuracy(device=model_device) for residue in residue_types}
        self.binary_precisions = {residue: BinaryPrecision(device=model_device) for residue in residue_types}
        self.binary_recalls = {residue: BinaryRecall(device=model_device) for residue in residue_types}
        self.binary_f1s = {residue: BinaryF1Score(device=model_device) for residue in residue_types}


    def configure_loss(self):
        def loss_function(pred_angle, angles, pred_seq, seqs, seq_loss_mask, angle_loss_mask):
            angle_loss = self.MSE(torch.cat([angles[...,:1],torch.sin(angles[...,1:3]), torch.cos(angles[...,1:3])],dim=-1),
            torch.cat([pred_angle[...,:1],torch.sin(pred_angle[...,1:3]), torch.cos(pred_angle[...,1:3])],dim=-1))
            
            angle_loss = angle_loss[angle_loss_mask].sum(dim=-1).mean()
            logits = pred_seq.permute(0,2,1)
            seq_loss = self.cross_entropy(logits, seqs)
            seq_loss = seq_loss[seq_loss_mask].mean()

            metric=Perplexity()
            metric.update(pred_seq[seq_loss_mask][None,...].cpu(), seqs[seq_loss_mask][None,...].cpu())
            perp = metric.compute()
            
            return {"angle_loss": angle_loss, "seq_loss": seq_loss, "perp":perp}

        self.loss_function = loss_function
        
    def load_model(self):
        params = OmegaConf.load(f'./src/models/configs/{self.hparams.model_name}.yaml')
        params.update(self.hparams)

        if self.hparams.model_name == 'GraphTrans':
            from src.models.graphtrans_model import GraphTrans_Model
            self.model = GraphTrans_Model(params)
        
        if self.hparams.model_name == 'StructGNN':
            from src.models.structgnn_model import StructGNN_Model
            self.model = StructGNN_Model(params)
            
        if self.hparams.model_name == 'GVP':
            from src.models.gvp_model import GVP_Model
            self.model = GVP_Model(params)

        if self.hparams.model_name == 'GCA':
            from src.models.gca_model import GCA_Model
            self.model = GCA_Model(params)

        if self.hparams.model_name == 'AlphaDesign':
            from src.models.alphadesign_model import AlphaDesign_Model
            self.model = AlphaDesign_Model(params)

        if self.hparams.model_name == 'ProteinMPNN':
            from src.models.proteinmpnn_model import ProteinMPNN_Model
            self.model = ProteinMPNN_Model(params)

        if self.hparams.model_name == 'ESMIF':
            pass

        if self.hparams.model_name == 'PiFold':
            from src.models.pifold_model import PiFold_Model
            self.model = PiFold_Model(params)

        if self.hparams.model_name == 'KWDesign':
            from src.models.kwdesign_model import Design_Model
            params['design_model'] = MemoPiFold_model(params)
            # self.model = Design_Model(params)
            self.model = Design_Model(
                params,
                temporature=params['temporature'],
                msa_n=params['msa_n'],
                tunning_layers_n=params['tunning_layers_n'],
                tunning_layers_dim=params['tunning_layers_dim'],
                input_design_dim=params['input_design_dim'],
                input_esm_dim=params['input_esm_dim'],
                tunning_dropout=params['tunning_dropout'],
                design_model=params['design_model'],
                LM_model=params['LM_model'],
                ESMIF_model=params['ESMIF_model'],
            )
        
        if self.hparams.model_name == 'E3PiFold':
            from src.models.E3PiFold_model import E3PiFold
            self.model = E3PiFold(params)

        if self.hparams.model_name == 'SurfProPiFold':
            from src.models.surfpropifold_model import SurfProPiFold_Model
            self.model = SurfProPiFold_Model(params)

        if self.hparams.model_name == 'SurfProPiFoldSurfaceOnly':
            from src.models.surfpropifold_model import SurfProPiFoldSurfaceOnly_Model
            self.model = SurfProPiFoldSurfaceOnly_Model(params)

        if self.hparams.model_name == 'SurfProPiFoldDense':
            from src.models.surfpropifold_model import SurfProPiFoldDense_Model
            self.model = SurfProPiFoldDense_Model(params)

        if self.hparams.model_name == 'TestModel0831':
            from src.models.test_models import TestModel0831
            self.model = TestModel0831(params)

        if self.hparams.model_name == 'TestModel0904':
            from src.models.test_models import TestModel0904
            self.model = TestModel0904(params)

        if self.hparams.model_name == 'TestModel0907':
            from src.models.test_models import TestModel0907
            self.model = TestModel0907(params)

        if self.hparams.model_name == 'SBModel':
            from src.models.SB_model import SBModel
            self.model = SBModel(params)
        if self.hparams.model_name == 'SBCModel':
            from src.models.SBC_model import SBCModel
            self.model = SBCModel(params)
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


    def gi(self):
        with torch.enable_grad():
            print((torch.tensor(0., requires_grad=True)*2).requires_grad)