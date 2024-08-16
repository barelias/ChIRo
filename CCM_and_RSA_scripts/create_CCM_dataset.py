import json
import traceback
import pandas as pd
import numpy as np
import progressbar
from typing import List
from rdkit import Chem
import numpy as np
from rdkit.Chem import Mol, AllChem
from cosymlib import Molecule, Geometry
from multiprocessing import Pool


def get_geo_symmetry(symmetry: str, coords: np.array, atom_ids: List[str], bonds: List[List[int]]):
    # Define geometry
    geometry = Geometry(positions=coords.tolist(),
                        symbols=atom_ids.tolist(),
                        connectivity=bonds)

    # Geometrical symmetry measure
    sym_geom_measure = geometry.get_symmetry_measure(symmetry, central_atom=1)
    return sym_geom_measure

def get_3d_geometry_with_atom_type_identifiers(mol):
    conf = mol.GetConformer()
    coords = np.array([conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())])
    atom_ids = np.array([atom.GetSymbol() for atom in mol.GetAtoms()])
    bonds = set()
    for i, atom in enumerate(Mol.GetAtoms(mol)):
        for bond in atom.GetBonds():
            indexes = [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]
            bonds.add((min(indexes)+1, max(indexes)+1))
    return coords, atom_ids, list(bonds)


def get_cs_symmetry_from_row(row):
    print (row['Index'])
    mol = row['rdkit_mol_cistrans_stereo']
    print (Chem.MolToSmiles(mol))
    try:
        symetry = get_geo_symmetry('Cs', *get_3d_geometry_with_atom_type_identifiers(mol))
        return symetry
    except Exception as exc:
        traceback.print_exc()
        return None

def process_chunk(chunk: pd.DataFrame):
    return chunk.apply(get_cs_symmetry_from_row, axis=1)

num_partitions = 10  # Number of partitions to split dataframe

test_final_RSA = pd.read_pickle("test_final_RSA.pkl")
train_final_RSA = pd.read_pickle("train_final_RSA.pkl")
validation_final_RSA = pd.read_pickle("validation_final_RSA.pkl")

total_number_of_conformers = test_final_RSA.shape[0] + train_final_RSA.shape[0] + validation_final_RSA.shape[0]

# Create a multiprocessing pool
pool = Pool(processes=num_partitions)

test_final_RSA['Index'] = test_final_RSA.index
train_final_RSA['Index'] = train_final_RSA.index
validation_final_RSA['Index'] = validation_final_RSA.index

test_final_RSA['CCM'] = pd.concat(pool.map(process_chunk, np.array_split(test_final_RSA, num_partitions)))
test_final_RSA.to_pickle('test_final_CCM.pkl')
validation_final_RSA['CCM'] = pd.concat(pool.map(process_chunk, np.array_split(validation_final_RSA, num_partitions)))
validation_final_RSA.to_pickle('validation_final_CCM.pkl')
train_final_RSA['CCM'] = pd.concat(pool.map(process_chunk, np.array_split(train_final_RSA, num_partitions)))
train_final_RSA.to_pickle('train_final_CCM.pkl')

# Close the pool
pool.close()
pool.join()
