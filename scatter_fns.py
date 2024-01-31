"""
Functions for geometric scattering feature extraction from
graphs: an implementation of Gao et al. 2019.

Citation:
Feng Gao, Guy Wolf, Matthew Hirn (2019). Geometric Scattering for Graph Data Analysis. 
Proceedings of the 36th International Conference on Machine Learning, PMLR 97:2122-2131. 
arXiv:1810.03068

"""


import utility_fns as uf
import numpy as np
import math
import networkx as nx
import scipy
import scipy.sparse as sparse
from scipy.stats import (skew, kurtosis)
from typing import List


def get_Dinv(graph: nx.Graph):
    """
    Computes the sparse inverse degree matrix.
    """
    n_nodes = graph.number_of_nodes()
    diag_ij = np.arange(n_nodes)
    d = np.array([1.0 / graph.degree[node] for node in graph.nodes])
    Dinv = sparse.coo_array((d, (diag_ij, diag_ij)), shape=(n_nodes, n_nodes))
    return Dinv #.toarray()


def get_P(graph: nx.Graph, Dinv: scipy.sparse.coo_array):
    """
    Computes P, the (n x n) lazy random walk matrix.
    P = 1/2*(I + A * Dinv), where A is the adjacency matrix 
    (symmetric/undirected; sparse).
    """
    n_nodes = graph.number_of_nodes()
    A = nx.adjacency_matrix(graph)
    P = 0.5 * (sparse.identity(n_nodes) + Dinv.dot(A))
    return P


def get_Psis(graph: nx.Graph, J: int = 5):
    """
    Computes wavelet matrices at scales 2^j for 1...J, where
    Psi_j = P^(2^(j-1)) - P^(2^j). Returns Psis in list.
    
    Ex: for J = 5:
    Psi_1 = P^1 - P^2
    ...
    Psi_5 = P^16 - P^32
    """
    Dinv = get_Dinv(graph)
    P = get_P(graph, Dinv)
    P_pwrs = [P]
    for j in range(1, J + 1):
        P *= P
        P_pwrs.append(P)
    Psi_l = [P_pwrs[i - 1] - P_pwrs[i] for i in range(1, J + 1)]
    return Psi_l


def get_raw_feat_matrix(graph: nx.Graph, 
                        feat_type: str = 'default',
                        **kwargs):
    """
    Computes node eccentricities and clustering coeffs for a graph
    and concatenates into a (n x f) matrix (where f is the number 
    of features), to be used as features for the geometric scattering 
    transform (see sect 4.1, p. 6).
    """
    n_nodes = graph.number_of_nodes()
    
    if feat_type == 'default':
        eccentricities = sparse.coo_array((n_nodes, 1)).toarray()
        clustering_coefs = sparse.coo_array((n_nodes, 1)).toarray()
        for i, node in enumerate(graph.nodes):
            eccentricities[i] = nx.eccentricity(graph, node)
            clustering_coefs[i] = nx.clustering(graph, node)
        raw_feat_matrix = np.column_stack((eccentricities, clustering_coefs))
        
    elif feat_type == 'dirac':
        from scipy.signal import unit_impulse 
        # k = np.max(kwargs['indexes']) + 1
        cols = [None] * len(kwargs['indexes'])
        for i, idx in enumerate(kwargs['indexes']):
            cols[i] = unit_impulse(n_nodes, idx)
        raw_feat_matrix = np.column_stack(cols)
        
    return raw_feat_matrix


def get_scat_mom_feat_v(raw_feat_matrix, 
                        Psi_l, 
                        Q: int = 4, 
                        normalize: bool = True,
                        feat_names = None):
    """
    Computes 0th, 1st, and 2nd order scattering moments (SMs) 
    for q \in 1...Q for graph features stored as columns in the
    `raw_feat_matrix`. Optionally normalized (e.g. using mean,
    variance, skew, and kurtosis).

    For now, this function assumes Q = 4. It has been updated to
    store scattering coefficients with labels in a dictionary.
    
    Reference equations are on p. 4 of Gao et al. 2019. 
    `f` is the number of features.
    
    quantity______scattering moments___reference
    f*Q           Zeroth-order         eq. 4
    f*J*Q         First-order          eq. 5
    f*(J c 2)*Q   Second-order         eq. 6
    """
    # store normalization fns by order (q)
    norm_fns = {
        1: np.mean,
        2: np.var,
        3: skew,
        4: kurtosis
    }
    n_feat = raw_feat_matrix.shape[1]
    feat_vs = [None] * n_feat
    J = len(Psi_l)

    # initialize dict to store scattering moments.  
        # heirarchy: feature index > SM order > list for qth moment stats
        # we initialize lists of predetermined sizes to prevent appends (faster in memory)
    feat_d = {}
    for f in range(n_feat):
        f_str = f'feat_{f}' if feat_names is None else feat_names[f]
        feat_d[f_str] = {
            '0th': {f'q{q}':{} for q in range(1, Q + 1)}, #[None] * Q,
            '1st': {f'q{q}':{} for q in range(1, Q + 1)}, #[None] * Q * J,
            '2nd': {f'q{q}':{} for q in range(1, Q + 1)} #[None] * Q * math.comb(J, 2)
        }
    
    # print('len(feat_d[0][\'2nd\']):', len(feat_d[0]['2nd']))

    # for each feature vector (col):
    for f in range(n_feat):
        f_str = f'feat_{f}' if feat_names is None else feat_names[f]
        # reset the insertion index counter for 2nd order values
        
        # 0th order SMs
        x = raw_feat_matrix[:, f]
        if normalize:
            for q in range(1, Q + 1):
                feat_d[f_str]['0th'][f'q{q}'] = norm_fns[q](x)
        else:
            # 0th order; q=1
            feat_d[f_str]['0th']['q_1'] = np.sum(x)
            # recursively calc SMs for powers of q>1
            x_copy = x.copy()
            for q in range(1, Q + 1):
                # recursively take x to powers of q
                x_copy = np.multiply(x_copy, x)
                feat_d[f_str]['0th'][f'q{q}'] = np.sum(x_copy)

        # 1st and 2nd order SMs
        for (j, Psi_j) in enumerate(Psi_l):

            # calc inner value of eq. 5 (also innermost of eq. 6)
            abs_Psij_x = np.abs(Psi_j.dot(x))
            
            if normalize:
                for q in range(1, Q + 1):
                    feat_d[f_str]['1st'][f'q{q}'][f'j{j+1}'] = norm_fns[q](abs_Psij_x)
                    for j_prime in range(j + 1, J):
                        # print(f"j:{j}, j\': {j_prime}, i_ctr_2nd:{i_ctr_2nd}")
                        jjpr_str = f'j{j+1}_jp{j_prime+1}'
                        feat_d[f_str]['2nd'][f'q{q}'][jjpr_str] = norm_fns[q](np.abs(Psi_l[j_prime].dot(abs_Psij_x)))
                        
            else:
                # 1st order; q=1
                feat_d[f_str]['1st']['q_1'][f'j{j}'] = np.sum(abs_Psij_x)

                # 2nd order; q=1
                for j_prime in range(j + 1, J):
                        jjpr_str = f'j{j+1}_jp{j_prime+1}'
                        feat_d[f_str]['2nd']['q_1'][jjpr_str] = np.sum(np.abs(Psi_l[j_prime].dot(abs_Psij_x)))
                
                # recursively calc SMs for powers of q>1
                abs_Psij_x_copy = abs_Psij_x.copy()
                for q in range(2, Q + 1):
                    
                    # 1st order; q>1
                    # recursively take abs_Psij_x to powers of q
                    abs_Psij_x_copy = np.multiply(abs_Psij_x_copy, abs_Psij_x)
                    feat_d[f_str]['1st'][f'q{q}'][f'j{j+1}'] = np.sum(abs_Psij_x_copy)
                    
                    # 2nd order; q>1
                    for j_prime in range(j + 1, J):
                        jjpr_str = f'j{j+1}_jp{j_prime+1}'
                        feat_d[f_str]['2nd'][f'q{q}'][jjpr_str] = np.sum(np.abs(Psi_l[j_prime].dot(abs_Psij_x)))

    return feat_d, uf.flatten_dict(feat_d)
                

def get_graphs_feat_matrix(graphs_l, 
                           Q: int = 4,
                           J: int = 5,
                           feat_type: str = 'default',
                           normalize_SMs: bool = True,
                           standardize_feats: bool = True,
                           **kwargs):
    """
    Uses all previous functions to generate the master
    (standardized) features matrix, for all features across
    all graphs, to feed into a classifier (e.g. SVM).
    """
    smfvs = [None] * len(graphs_l)
    for (i, graph) in enumerate(graphs_l):
        # print(f"graph {i}")
        Psi_l = get_Psis(graph, J)
        raw_feat_matrix = get_raw_feat_matrix(graph=graph,
                                              feat_type=feat_type,
                                              **kwargs)
        feat_d, smfv = get_scat_mom_feat_v(raw_feat_matrix,
                                           Psi_l,
                                           Q,
                                           normalize_SMs)
        smfvs[i] = smfv

    # combine graphs' SM feature vects such that graphs are rows
    smfvs = np.vstack(smfvs)

    # standardize (0-1 scale) the smfvs matrix feature-wise
    if standardize_feats:
        smfvs = (smfvs - smfvs.min(axis=0)) / smfvs.ptp(axis=0)

    return smfvs














    
    
## OLD
# def get_scat_mom_feat_v(raw_feat_matrix, 
#                         Psi_l, 
#                         Q: int = 4, 
#                         normalize: bool = True,
#                         return_dict: bool = False):
#     """
#     Computes 0th, 1st, and 2nd order scattering moments (SMs) 
#     for q \in 1...Q for graph features stored as columns in the
#     `raw_feat_matrix`. Optionally normalized.

#     For now, this function assumes Q = 4.
    
#     Reference equations are on p. 4 of Gao et al. 2019. 
#     `f` is the number of features.
    
#     quantity______scattering moments___reference
#     f*Q           Zeroth-order         eq. 4
#     f*J*Q         First-order          eq. 5
#     f*(J c 2)*Q   Second-order         eq. 6
#     """
#     # store normalization fns by order (q)
#     norm_fns = {
#         1: np.mean,
#         2: np.var,
#         3: skew,
#         4: kurtosis
#     }
#     n_feat = raw_feat_matrix.shape[1]
#     feat_vs = [None] * n_feat
#     J = len(Psi_l)

#     # initialize dict to store scattering moments.  
#         # heirarchy: feature index > SM order > list for qth moment stats
#         # we initialize lists of predetermined sizes to prevent appends (faster in memory)
#     feat_d = {}
#     for f in range(n_feat):
#         feat_d[f_str] = {
#             '0th': [None] * Q,
#             '1st': [None] * Q * J,
#             '2nd': [None] * Q * math.comb(J, 2)
#         }
    
#     # print('len(feat_d[0][\'2nd\']):', len(feat_d[0]['2nd']))

#     # for each feature vector (col):
#     for f in range(n_feat):
#         # reset the insertion index counter for 2nd order values
#         i_ctr_2nd = 0
        
#         # 0th order SMs
#         x = raw_feat_matrix[:, f]
#         if normalize:
#             for q in range(1, Q + 1):
#                 feat_d[f_str]['0th'][q - 1] = norm_fns[q](x)
#         else:
#             # 0th order; q=1
#             feat_d[f_str]['0th'][0] = np.sum(x)
#             # recursively calc SMs for powers of q>1
#             x_copy = x.copy()
#             for q in range(1, Q + 1):
#                 # recursively take x to powers of q
#                 x_copy = np.multiply(x_copy, x)
#                 feat_d[f_str]['0th'][q - 1] = np.sum(x_copy)

#         # 1st and 2nd order SMs
#         for (j, Psi_j) in enumerate(Psi_l):

#             # calc inner value of eq. 5 (also innermost of eq. 6)
#             abs_Psij_x = np.abs(Psi_j.dot(x))
            
#             if normalize:
#                 for q in range(1, Q + 1):
#                     idx_1st = (j - 1) * Q + (q - 1)
#                     feat_d[f_str]['1st'][idx_1st] = norm_fns[q](abs_Psij_x)
#                     for j_prime in range(j + 1, J):
#                         # print(f"j:{j}, j\': {j_prime}, i_ctr_2nd:{i_ctr_2nd}")
#                         feat_d[f_str]['2nd'][i_ctr_2nd] = norm_fns[q](np.abs(Psi_l[j_prime].dot(abs_Psij_x)))
#                         i_ctr_2nd += 1
#             else:
#                 # 1st order; q=1
#                 feat_d[f_str]['1st'][(j - 1) * Q] = np.sum(abs_Psij_x)
                
#                 # recursively calc SMs for powers of q>1
#                 abs_Psij_x_copy = abs_Psij_x.copy()
#                 for q in range(2, Q + 1):
                    
#                     # 1st order; q>1
#                     # recursively take abs_Psij_x to powers of q
#                     abs_Psij_x_copy = np.multiply(abs_Psij_x_copy, abs_Psij_x)
#                     feat_d[f_str]['1st'][(j - 1) * Q + q - 1] = np.sum(abs_Psij_x_copy)
                    
#                     # 2nd order; q>1
#                     for j_prime in range(j + 1, J):
#                         feat_d[f_str]['2nd'][i_ctr_2nd] = np.sum(np.abs(Psi_l[j_prime].dot(abs_Psij_x)))
#                         i_ctr_2nd += 1

#         feat_vs[f] = np.concatenate((feat_d[f_str]['0th'], 
#                                      feat_d[f_str]['1st'], 
#                                      feat_d[f_str]['2nd']))

#     if return_dict:
#         return feat_d
#     else:
#         # return a single vector of stacked SMs by feature
#         graph_feat_v = np.concatenate(feat_vs)
#         return graph_feat_v