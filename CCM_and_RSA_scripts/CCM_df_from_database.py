from functools import partial
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

client = MongoClient(host="localhost", port=27017, username="root", password="MongoDB2019!")
collection = client.ttc.ccm

test_final_RSA = pd.read_pickle("./test_final_RSA.pkl")
train_final_RSA = pd.read_pickle("./train_final_RSA.pkl")
validation_final_RSA = pd.read_pickle("./validation_final_RSA.pkl")

test_final_RSA['Index'] = test_final_RSA.index
train_final_RSA['Index'] = train_final_RSA.index
validation_final_RSA['Index'] = validation_final_RSA.index

def get_CCM_from_db(row, df_type):
    global collection
    mol = collection.find_one({
        'Index': row['Index'],
        'df_type': df_type
    })
    if mol is None: return None
    return mol['CCM']

def verify_CCM(row):
    return row['CCM'] != None

test_final_RSA['CCM'] = test_final_RSA.apply(partial(get_CCM_from_db, df_type='test'), axis=1)
train_final_RSA['CCM'] = train_final_RSA.apply(partial(get_CCM_from_db, df_type='train'), axis=1)
validation_final_RSA['CCM'] = validation_final_RSA.apply(partial(get_CCM_from_db, df_type='validation'), axis=1)

test_final_RSA = test_final_RSA[test_final_RSA.apply(verify_CCM, axis=1)]
train_final_RSA = train_final_RSA[train_final_RSA.apply(verify_CCM, axis=1)]
validation_final_RSA = validation_final_RSA[validation_final_RSA.apply(verify_CCM, axis=1)]

test_final_RSA = test_final_RSA.reset_index(drop=True)
train_final_RSA = train_final_RSA.reset_index(drop=True)
validation_final_RSA = validation_final_RSA.reset_index(drop=True)

test_final_RSA.to_pickle('test_final_CCM.pkl')
train_final_RSA.to_pickle('train_final_CCM.pkl')
validation_final_RSA.to_pickle('validation_final_CCM.pkl')