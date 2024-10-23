from functools import wraps
import signal
import traceback
import pandas as pd
import numpy as np
from typing import Callable, List
from rdkit import Chem
import numpy as np
from rdkit.Chem import Mol, AllChem
from cosymlib import Molecule, Geometry
from multiprocessing import Pool
from multiprocessing.context import TimeoutError
from pymongo import MongoClient

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
    mol = row['rdkit_mol_cistrans_stereo']
    print (Chem.MolToSmiles(mol))
    try:
        symetry = get_geo_symmetry('Cs', *get_3d_geometry_with_atom_type_identifiers(mol))
        return symetry
    except Exception as exc:
        traceback.print_exc()
        return None

client = MongoClient(host="localhost", port=27017, username="root", password="MongoDB2019!")
collection = client.ttc.ccm
num_partitions = 1  # Number of partitions to split dataframe

test_final_RSA = pd.read_pickle("./test_final_RSA.pkl")
train_final_RSA = pd.read_pickle("./train_final_RSA.pkl")
validation_final_RSA = pd.read_pickle("./validation_final_RSA.pkl")

total_number_of_conformers = test_final_RSA.shape[0] + train_final_RSA.shape[0] + validation_final_RSA.shape[0]

test_final_RSA['Index'] = test_final_RSA.index
train_final_RSA['Index'] = train_final_RSA.index
validation_final_RSA['Index'] = validation_final_RSA.index

def process_df(df, pool_size, timeout, collection, df_type):
    
    for i in range(0, len(df), pool_size):
        with Pool(processes=pool_size) as pool:
            df_block = df.iloc[i:i + pool_size]
            to_insert_pre_result = []
            
            futures = []
            for _, rows in df_block.iterrows():
                mol = collection.find_one({
                    'Index': rows['Index'],
                    'df_type': df_type
                })
                if mol is None:
                    futures.append(pool.apply_async(get_cs_symmetry_from_row, (rows,)))
                    to_insert_pre_result.append({
                        'Index': rows['Index'],
                        'ID': rows['ID'],
                        'df_type': df_type,
                        'SMILES_nostereo': rows['SMILES_nostereo'],
                        'CCM': None
                    })
                else:
                    print ('exists')
            for idx, future in enumerate(futures):
                try:
                    if timeout != -1:
                        sym = future.get(timeout=timeout)
                    else:
                        sym = future.get()
                    to_insert = to_insert_pre_result[idx]
                    to_insert['CCM'] = sym
                    collection.insert_one(to_insert)
                except TimeoutError:
                    print (f'timeout {timeout}')
                
            
for timeout in (1, 3, 5, 10, -1):
    process_df(test_final_RSA, 4, timeout, collection, 'test')
    process_df(validation_final_RSA, 4, timeout, collection, 'validation')
    process_df(train_final_RSA, 4, timeout, collection, 'train')
    
    # test_final_RSA['CCM'] = pd.concat(pool.map(process_chunk, np.array_split(test_final_RSA, num_partitions)))
    # test_final_RSA['CCM'] = process_chunk(test_final_RSA)
    # test_final_RSA.to_pickle('test_final_CCM.pkl')
    # validation_final_RSA['CCM'] = pd.concat(pool.map(process_chunk, np.array_split(validation_final_RSA, num_partitions)))
    # validation_final_RSA['CCM'] = process_chunk(validation_final_RSA)
    # validation_final_RSA.to_pickle('validation_final_CCM.pkl')
    # train_final_RSA['CCM'] = pd.concat(pool.map(process_chunk, np.array_split(train_final_RSA, num_partitions)))
    # train_final_RSA['CCM'] = process_chunk(train_final_RSA)
    # train_final_RSA.to_pickle('train_final_CCM.pkl')
