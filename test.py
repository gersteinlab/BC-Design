import os
from Bio import PDB

def validate_and_save_pdb(pdb_file, output_file):
    """
    Validates and resaves a PDB file.
    
    :param pdb_file: Path to the input PDB file
    :param output_file: Path to the output PDB file
    """
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('structure', pdb_file)
    
    io = PDB.PDBIO()
    io.set_structure(structure)
    io.save(output_file)

def resave_all_pdbs(source_dir, target_dir):
    """
    Iterates over all PDB files in the source directory and resaves them to the target directory.
    
    :param source_dir: Path to the directory containing the original PDB files
    :param target_dir: Path to the directory where the validated PDB files will be saved
    """
    # Ensure the target directory exists
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Iterate over all files in the source directory
    for filename in os.listdir(source_dir):
        if filename.endswith(".pdb"):  # Process only PDB files
            input_file = os.path.join(source_dir, filename)
            output_file = os.path.join(target_dir, filename)
            
            print(f"Processing {input_file} -> {output_file}")
            try:
                validate_and_save_pdb(input_file, output_file)
            except Exception as e:
                print(f"Failed to process {input_file}: {e}")

# Define source and target directories
source_dir = "/gpfs/gibbs/pi/gerstein/xt86/surface/ProteinInvBench-lightning/predicted_pdb/SBC2-sum3-minlrdiv1-bs4-lr00002-epoch20-test-old/CATH4.2SurfProPiFoldDense"
target_dir = "/gpfs/gibbs/pi/gerstein/xt86/surface/ProteinInvBench-lightning/predicted_pdb/SBC2-sum3-minlrdiv1-bs4-lr00002-epoch20-test/CATH4.2SurfProPiFoldDense"

# Resave all PDB files from the source directory to the target directory
resave_all_pdbs(source_dir, target_dir)
