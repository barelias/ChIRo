import pandas as pd
import networkx as nx
import rdkit

def get_all_paths(G, N = 3):
    # adapted from: https://stackoverflow.com/questions/28095646/finding-all-paths-walks-of-given-length-in-a-networkx-graph
    
    def findPaths(G,u,n):
        if n==0:
            return [[u]]
        paths = [[u]+path for neighbor in G.neighbors(u) for path in findPaths(G,neighbor,n-1) if u not in path]
        return paths
    
    allpaths = []
    for node in G:
        allpaths.extend(findPaths(G,node,N))
    
    return allpaths

def verify_if_has_dihedral_angle_or_more(row):
    adj = rdkit.Chem.GetAdjacencyMatrix(row['rdkit_mol_cistrans_stereo'])
    graph = nx.from_numpy_array(adj, parallel_edges=False, create_using=None)
    distance_paths, angle_paths, dihedral_paths = get_all_paths(graph, N = 1), get_all_paths(graph, N = 2), get_all_paths(graph, N = 3)
    
    return len(dihedral_paths) != 0
        
    # return row['rdkit_mol_cistrans_stereo'].GetNumAtoms() >= 6

# df[df.apply(lambda row: len(row['Name']) > 4, axis=1)]

test_final_RSA = pd.read_pickle("test_final_RSA_class.pkl")
train_final_RSA = pd.read_pickle("train_final_RSA_class.pkl")
validation_final_RSA = pd.read_pickle("validation_final_RSA_class.pkl")

filtered_test_final_RSA = test_final_RSA[test_final_RSA.apply(verify_if_has_dihedral_angle_or_more, axis=1)]
filtered_train_final_RSA = train_final_RSA[train_final_RSA.apply(verify_if_has_dihedral_angle_or_more, axis=1)]
filtered_validation_final_RSA = validation_final_RSA[validation_final_RSA.apply(verify_if_has_dihedral_angle_or_more, axis=1)]

print(filtered_test_final_RSA.shape[0])
print(filtered_train_final_RSA.shape[0])
print(filtered_validation_final_RSA.shape[0])

filtered_test_final_RSA = filtered_test_final_RSA.reset_index(drop=True)
filtered_train_final_RSA = filtered_train_final_RSA.reset_index(drop=True)
filtered_validation_final_RSA = filtered_validation_final_RSA.reset_index(drop=True)

filtered_test_final_RSA.to_pickle('test_final_RSA_class.pkl')
filtered_train_final_RSA.to_pickle('train_final_RSA_class.pkl')
filtered_validation_final_RSA.to_pickle('validation_final_RSA_class.pkl')