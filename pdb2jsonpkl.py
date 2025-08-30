from Bio import PDB
import os
import json
import numpy as np
import pickle
import torch
from tqdm import tqdm
from Bio.PDB import PDBParser, Structure, Model, Chain, Residue, Atom
from Bio.PDB.ResidueDepth import get_surface
from scipy.spatial import cKDTree, Delaunay
from Bio.SeqUtils import seq1
from Bio.PDB.Polypeptide import is_aa
from Bio.PDB.PDBExceptions import PDBConstructionWarning
import warnings
import argparse # Import the argparse library

# Suppress PDBConstructionWarning
warnings.simplefilter('ignore', PDBConstructionWarning)

# Define MSMS executable path
msms_exec = '/gpfs/gibbs/pi/gerstein/xt86/surface/msms/msms.x86_64Linux2.2.6.1'
os.chmod(msms_exec, 0o755)

# Define the biochemical features dictionary
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

# Mapping from three-letter codes to one-letter codes
three_to_one = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F", "GLY": "G",
    "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L", "MET": "M", "ASN": "N",
    "PRO": "P", "GLN": "Q", "ARG": "R", "SER": "S", "THR": "T", "VAL": "V",
    "TRP": "W", "TYR": "Y"
}


def parse_pdb(file_path):
    name = os.path.basename(file_path).replace('.pdb', '')
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure(name, file_path)

    seq = ''
    coords = []

    for model in structure:
        for chain in model:
            seq += ''.join([three_to_one[res.get_resname()] for res in chain if res.get_id()[0] == ' '])
            # temp_coords = []
            atom_names = ['N', 'CA', 'C', 'O']

            for res in chain:
                if res.get_resname() in three_to_one.keys():
                    coord_dict = {atom.get_name(): atom.get_coord().tolist() for atom in res if atom.get_name() in atom_names}
                    if all(atom in coord_dict for atom in atom_names):  # Ensure all atoms are present
                        # temp_coords.append([coord_dict[atom] for atom in atom_names])
                        temp_coords = [coord_dict[atom] for atom in atom_names]
                        if len(temp_coords) == 4:  # Collect 4 sets of coordinates
                            coords.append(temp_coords)
                            # temp_coords = []

    return {'name': name, 'seq': seq, 'coords': coords}


# Step 1: Create PDB structure from protein dict
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
            atom = Atom.Atom(atom_name, coord, 1.0, 0.0, ' ', atom_name, atom_index, atom_name[0])
            residue.add(atom)
        chain.add(residue)
    
    model.add(chain)
    structure.add(model)
    return structure

# Step 2: Feature assignment
# Function to get atom coordinates and residue types
def get_atom_coords_and_residues(structure):
    coords = []
    residues = []
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    coords.append(atom.coord)
                    residues.append(three_to_one.get(residue.get_resname(), ''))
    return np.array(coords), residues


# Process each PDB file in the input directory
def assign_features(surface, structure):
    atom_coords, residue_types = get_atom_coords_and_residues(structure)
    
    # Build k-D tree for atom coordinates
    kdtree = cKDTree(atom_coords)
    
    # Assign biochemical features to each vertex in the surface
    features = []
    for vertex in surface:
        dist, idx = kdtree.query(vertex)
        residue_type = residue_types[idx]
        residue_features = [bio_feat_dict[feat].get(residue_type, 0) for feat in bio_feat_dict]
        features.append(residue_features)
    
    # Convert features to a numpy array
    features_array = np.array(features)
    
    return features_array


# Step 3: Smooth the surface
# Function to perform Gaussian kernel smoothing on all points using PyTorch
def gaussian_kernel_smoothing(coords, k=8, eta=None):
    # print(coords.shape)
    if len(coords) > 20000:
        # Generate random permutation of indices
        indices = torch.randperm(len(coords))[:20000]
        # Select the random indices along the 0-th axis
        coords = coords[indices]
    # Convert numpy array to PyTorch tensor and move to GPU
    coords = torch.tensor(coords, dtype=torch.float32).cuda(0)
    
    # Compute the full pairwise distance matrix
    dists = torch.cdist(coords, coords, p=2)
    
    if eta is None:
        eta = torch.max(dists).item()
    
    nearest_neighbors = torch.argsort(dists, dim=1)[:, 1:k+1]
    
    # Get the distances of the k-nearest neighbors
    nearest_dists = torch.gather(dists, 1, nearest_neighbors)
    
    # Compute weights using the Gaussian kernel
    weights = torch.exp(-nearest_dists**2 / eta)
    weights /= torch.sum(weights, dim=1, keepdim=True)
    
    # Compute the smoothed coordinates
    smoothed_coords = torch.sum(weights[:, :, None] * coords[nearest_neighbors], dim=1)
    
    return smoothed_coords.cpu().numpy()


# Step 4: Compress the surface and features using octree-based compression
class OctreeNode:
    def __init__(self, points, indices):
        self.points = points
        self.indices = indices
        self.children = []

def create_octree(points, indices, min_points_per_cube):
    """
    Create an octree for the given points.
    """
    def divide(points, indices):
        if len(points) <= min_points_per_cube:
            return OctreeNode(points, indices)
        centroid = np.mean(points, axis=0)
        partitions = [[] for _ in range(8)]
        partition_indices = [[] for _ in range(8)]
        for idx, point in enumerate(points):
            partition_index = 0
            if point[0] > centroid[0]:
                partition_index += 1
            if point[1] > centroid[1]:
                partition_index += 2
            if point[2] > centroid[2]:
                partition_index += 4
            partitions[partition_index].append(point)
            partition_indices[partition_index].append(indices[idx])
        node = OctreeNode(None, None)
        node.children = [divide(part, part_idx) for part, part_idx in zip(partitions, partition_indices)]
        return node

    return divide(points, indices)

def gather_points(node):
    """
    Gather points and indices from the octree.
    """
    if node.points is not None:
        return [(node.points, node.indices)]
    result = []
    for child in node.children:
        result.extend(gather_points(child))
    return result

def compress_surface(points, features, down_sample_ratio, min_points_per_cube=32):
    """
    Compress the surface and features using octree-based compression.
    """
    indices = np.arange(points.shape[0])
    octree = create_octree(points, indices, min_points_per_cube)
    compressed_points = []
    compressed_features = []
    for cube_points, cube_indices in gather_points(octree):
        local_density = len(cube_points)
        num_points = int(local_density * down_sample_ratio)
        if num_points > 0:
            sampled_indices = np.random.choice(local_density, num_points, replace=False)
            sampled_points = np.array(cube_points)[sampled_indices]
            sampled_features = features[cube_indices][sampled_indices]
            compressed_points.extend(sampled_points)
            compressed_features.extend(sampled_features)
    return np.array(compressed_points), np.array(compressed_features)


# Step 5: Add interior points
# Function to get biochemical features from a residue
def get_biochem_features(residue):
    # Define hydrophobicity scale (Kyte-Doolittle)
    hydrophobicity_scale = {
        'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
        'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
        'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
        'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
    }
    
    # Define charge scale
    charge_scale = {
        'D': -1, 'E': -1, 'K': 1, 'R': 1, 'H': 0.1  # Histidine is partially charged
    }
    
    # Define polarity, acceptor, and donor features as shown in the image
    polarity_scale = {'R': 1, 'N': 1, 'D': 1, 'Q': 1, 'E': 1, 'H': 1, 'K': 1, 'S': 1, 'T': 1, 'Y': 1}
    acceptor_scale = {'D': 1, 'E': 1, 'N': 1, 'Q': 1, 'H': 1, 'S': 1, 'T': 1, 'Y': 1}
    donor_scale = {'R': 1, 'K': 1, 'W': 1, 'N': 1, 'Q': 1, 'H': 1, 'S': 1, 'T': 1, 'Y': 1}
    
    res_3letter = residue.get_resname()  # Get the three-letter code
    res_1letter = seq1(res_3letter)  # Convert to one-letter code
    hydrophobicity = hydrophobicity_scale.get(res_1letter, 0)
    charge = charge_scale.get(res_1letter, 0)
    polarity = polarity_scale.get(res_1letter, 0)
    acceptor = acceptor_scale.get(res_1letter, 0)
    donor = donor_scale.get(res_1letter, 0)
    return np.array([hydrophobicity, charge, polarity, acceptor, donor])


def add_interior_points(surface_points, surface_features, structure):
    # Extract residue info and calculate biochemical features
    coords = []
    features = []

    for model in structure:
        for chain in model:
            for residue in chain:
                if is_aa(residue) and 'CA' in residue:
                    res_coord = residue['CA'].get_coord()  # Get alpha carbon coordinates
                    res_features = get_biochem_features(residue)

                    coords.append(res_coord)
                    features.append(res_features)

    coords = np.array(coords)
    features = np.array(features)

    # Build a KDTree for fast nearest-neighbor search
    kdtree = cKDTree(coords)

    # Generate random points inside the surface
    min_coords = surface_points.min(axis=0)
    max_coords = surface_points.max(axis=0)
    num_samples = 5000
    random_points = np.random.uniform(min_coords, max_coords, (num_samples, 3))

    tri = Delaunay(surface_points)

    def is_inside(point, tri):
        return tri.find_simplex(point) >= 0

    inside_points = np.array([p for p in random_points if is_inside(p, tri)])

    # Assign biochemical features to random points based on nearest residue
    _, idx = kdtree.query(inside_points)
    inside_features = features[idx]

    # Concatenate surface and inside points and features
    new_surface = np.concatenate([surface_points, inside_points], axis=0)
    new_features = np.concatenate([surface_features, inside_features], axis=0)

    return new_surface, new_features


# Step 6: Sample
def sample_if_needed(data_dict, max_length=5000):
    for key, value in data_dict.items():
        surface = value['surface']
        features = value['features']
        
        if len(surface) > max_length:
            indices = np.random.choice(len(surface), max_length, replace=False)
            value['surface'] = surface[indices]
            value['features'] = features[indices]
    
    return data_dict


# Main function to run the pipeline
def main(dataset='afdb2000'):
    input_json_path = f'data/{dataset}/{dataset}.json'
    output_pkl_path = f'data/{dataset}/{dataset}.pkl'

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_pkl_path)
    os.makedirs(output_dir, exist_ok=True)
    
    with open(input_json_path, 'r') as f:
        protein_dicts = json.load(f)
    
    combined_data = {}
    # test_i = 0
    for protein_data in tqdm(protein_dicts, desc="Processing proteins"):
        # if test_i < 4:
        #     test_i += 1
        #     continue
        structure = create_pdb_structure(protein_data)
        # print('done struc')
        # print(len(protein_data['seq']))
        # print(len(protein_data['coords']))
        try:
            surface = get_surface(structure[0], MSMS=msms_exec)
        except Exception as e:
            print(f"Failed to generate surface for {protein_data['name']}: {e}")
            continue
        # print('done surface')
        features = assign_features(surface, structure)
        # print('done features')
        # Step 3: Smooth the surface
        smoothed_surface = gaussian_kernel_smoothing(surface)
        # print('done smooth')
        # Step 4: Compress the surface and features using octree-based compression
        if len(smoothed_surface) > 5000:
            down_sample_ratio = 5000 / len(smoothed_surface)
            compressed_points, compressed_features = compress_surface(smoothed_surface, features, down_sample_ratio)
        else:
            compressed_points, compressed_features = smoothed_surface, features  # No down-sampling
        # print('done compress')
        # Step 5: Add interior points
        final_surface, final_features = add_interior_points(compressed_points, compressed_features, structure)
        
        combined_data[protein_data['name']] = {
            'surface': final_surface,
            'features': final_features,
            'seq': protein_data['seq']
        }

    combined_data = sample_if_needed(combined_data)

    # Save the final data into a .pkl file
    with open(output_pkl_path, 'wb') as f:
        pickle.dump(combined_data, f)
    return combined_data




def get_sequence_from_pdb(pdb_file_path):
    """
    Parses a PDB file and returns the amino acid sequence of the first chain.
    
    Args:
        pdb_file_path (str): The full path to the PDB file.

    Returns:
        str: The one-letter amino acid sequence, or None if parsing fails.
    """
    try:
        parser = PDBParser()
        structure = parser.get_structure("gt_structure", pdb_file_path)
        
        # Extract sequence from the first model and first chain
        model = next(structure.get_models())
        chain = next(model.get_chains())
        
        sequence = "".join(
            three_to_one.get(residue.get_resname(), 'X') 
            for residue in chain 
            if is_aa(residue)
        )
        return sequence
    except Exception as e:
        print(f"Warning: Could not parse sequence from {pdb_file_path}. Error: {e}")
        return None


if __name__ == "__main__":
    # Define paths for predicted and ground truth PDBs
    # pdb_folder = './data/antonia0824/pdbs'
    pdb_folder = './data/antonia0830/pdbs'
    
    pdb_files = [f for f in os.listdir(pdb_folder) if f.endswith('.pdb')]
    data = []

    print("--- Creating initial dataset ---")
    for pdb_file in tqdm(pdb_files, desc="Parsing PDBs"):
        predicted_path = os.path.join(pdb_folder, pdb_file)

        # Call the new function that combines data from two PDBs
        combined_data = parse_pdb(predicted_path)
        
        if combined_data:
            data.append(combined_data)

    # --- The rest of the script remains the same ---

    # Create dataset name from the folder path
    # dataset_name = 'antonia0824'
    dataset_name = 'antonia0830'

    # Create and save the initial JSON file
    output_data_dir = os.path.join('./data', dataset_name)
    os.makedirs(output_data_dir, exist_ok=True)
    json_output_path = os.path.join(output_data_dir, dataset_name + '.json')

    with open(json_output_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)
    print(f"\nInitial JSON saved to: {json_output_path}")

    # Call the main function to process JSON and create PKL
    main(dataset_name)
    
