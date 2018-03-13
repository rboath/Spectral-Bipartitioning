import networkx as nx
import numpy as np
from numpy import linalg as LA

# We first define a function to return the second smallest value in a list

def Second_Smallest(values):
    (n_1, n_2) = (float('inf'), float('inf'))
    for x in values:
        if x <= n_1:
            (n_1, n_2) = (x, n_1)
        elif x < n_2:
            n_2 = x
    return n_2
    
# We now define our spectral bipartition function, which takes as inputs a networkx graph,
# and an optional argument of the desired size of one of the clusters (which then fixes
# the size of the second cluster). If not specified, we make the clusters equal size (plus
# or minus 1). The spectral bipartitioning method returns the partition with the fewest
# number of edges between the two clusters.

def Spectral_Bipartition(G, n_1="temp"):     # Since the size of one of the sets in the partition fixes
    n = nx.number_of_nodes(G)                # the size of the other, we only use one as an input.
    if n_1 == "temp":                        # If partition size is not specified, function returns two 
        n_1 = int(np.floor(n/2))             # clusters with equal sizes (plus or minus 1)
    n_2 = n - n_1
    A = nx.adjacency_matrix(G)
    Deg = [d for n, d in G.degree()]
    Diag = np.diag(Deg)
    Laplacian = Diag - A
    eigvals, eigvecs = LA.eig(Laplacian)
    Fiedler_val = Second_Smallest(eigvals)
    i = np.where(eigvals == Fiedler_val)
    Fiedler_vec = eigvecs[:,i]
    Fiedler_vec_list = Fiedler_vec.tolist()
    Flat_1 = [x for y in Fiedler_vec_list for x in y]
    Flat_Fiedler_vec = [x for y in Flat_1 for x in y]
    Fiedler_vec_array = np.array(Flat_Fiedler_vec)
    ind1 = (np.argpartition(Fiedler_vec_array, -n_1)[-n_1:])
    s_1 = np.ones(n)
    for i in ind1:
        s_1[i] = s_1[i] - 2
    ind2 = (np.argpartition(Fiedler_vec_array, -n_2)[-n_2:])
    s_2 = np.ones(n)
    for i in ind2:
        s_2[i] = s_2[i] - 2
    R_1 = 0.25*np.dot(((s_1)*Laplacian),s_1)
    R_2 = 0.25*np.dot(((s_2)*Laplacian),s_2)
    if R_1 <= R_2:
        print("Cut Size:")
        print(R_1[0,0])
        Group_1 = np.empty(0)
        Group_2 = np.empty(0)
        for i in range(0,n):
            if s_1[i] == 1:
                Group_1 = np.append(Group_1, [i], axis=0)
            else:
                Group_2 = np.append(Group_2, [i], axis=0)
        print("Nodes in community 1:")
        print(Group_1)
        print("Nodes in community 2:")
        print(Group_2)
    else:
        print("Cut Size:")
        print(R_2[0,0])
        Group_1 = np.empty(0)
        Group_2 = np.empty(0)
        for i in range(0,n):
            if s_2[i] == 1:
                Group_1 = np.append(Group_1, [i], axis=0)
            else:
                Group_2 = np.append(Group_2, [i], axis=0)
        print("Nodes in community 1:")
        print(Group_1)
        print("Nodes in community 2:")
        print(Group_2)
