from apted import APTED, Config

class MyNode:
    def __init__(self, label, children=None):
        self.label = label
        self.children = children or []

    def get_children(self):
        return self.children

class SimpleConfig(Config):
    def rename(self, node1, node2):
        return 0 if node1.label == node2.label else 1
    def insert(self, node):
        return 1
    def delete(self, node):
        return 1

def build_my_tree(tree, index):
    children = [build_my_tree(tree, c) for c in tree.successors.get(index, [])]
    return MyNode(str(tree.node_labels[index]), children)

def fast_ted(tree1, tree2):
    t1 = build_my_tree(tree1, tree1.root)
    t2 = build_my_tree(tree2, tree2.root)
    apted = APTED(t1, t2, SimpleConfig())
    return apted.compute_edit_distance()





from scipy.sparse import csgraph, lil_matrix
from scipy.sparse.linalg import eigsh
import numpy as np

def tree_to_sparse_adj_matrix(tree):
    n = len(tree)
    A = lil_matrix((n, n))
    for i, succs in tree.successors.items():
        for j in succs:
            A[i, j] = 1
            A[j, i] = 1
    return A.tocsr()

def spectral_distance(tree1, tree2, k=20):

    A1 = tree_to_sparse_adj_matrix(tree1)
    #print("A1")
    A2 = tree_to_sparse_adj_matrix(tree2)
    #print("A2")

    L1 = csgraph.laplacian(A1, normed=True)
    #print("L1")
    L2 = csgraph.laplacian(A2, normed=True)
    #print("L2")

    eigs1 = np.sort(eigsh(L1, k=k, which='SM', return_eigenvectors=False))
    #print("eigs1")
    eigs2 = np.sort(eigsh(L2, k=k, which='SM', return_eigenvectors=False))
    #print("eigs2")

    return np.linalg.norm(eigs1 - eigs2)



