"""
Function(s) for importing and processing the IMDB-BINARY
graphs dataset. Uses the graph class of `networkx` to store
the graphs conveniently.

Refs for the IMDB-BINARY dataset:
original creator: https://dl.acm.org/doi/10.1145/2783258.2783417
https://networkrepository.com/IMDB-BINARY.php
https://paperswithcode.com/dataset/imdb-binary
"""


import numpy as np
import pandas as pd
import networkx as nx


def import_IMDBB_graphs(path: str = './IMDB-BINARY',
                        return_df: bool = True):
    """
    This imports the HuggingFace JSON version of the
    IMDB-BINARY dataset.
    https://huggingface.co/datasets/graphs-datasets/IMDB-BINARY
    """

    def create_graph(edge_ll):
        """
        Converts a double-lists for edges in the HuggingFace
        dataset to a networkx Graph object.
        """
        g = nx.Graph()
        edges = [None] * len(edge_ll[0])
        for i, node in enumerate(edge_ll[0]):
            edges[i] = (edge_ll[0][i], edge_ll[1][i])
        g.add_edges_from(edges)
        return g
        
    df = pd.read_json(path_or_buf=f"{path}/IMDB-BINARY.jsonl", lines=True)
    df['graph'] = df['edge_index'].apply(create_graph)
    df.drop(['edge_index'], axis=1, inplace=True)
    df['y'] = df['y'].apply(lambda x: x[0])

    if return_df:
        return df
    else:
        return (df['graph'], df['y'])


def get_IMDBB_graphs_stats(graphs_l, labels):
    """
    Calculates descriptive stats for the IMDB-Binary
    dataset.
    """
    print('IMDB-Binary descriptive stats:')
    print(f'\nnum. graphs = {len(graphs_l)}')
    n_nodes_l = [g.number_of_nodes() for g in graphs_l]
    min_n_nodes = np.min(n_nodes_l)
    max_n_nodes = np.max(n_nodes_l)
    median_n_nodes = np.median(n_nodes_l)
    print('\nnum. nodes:')
    print('-----------')
    print(f'min     {min_n_nodes}')
    print(f'max     {max_n_nodes}')
    print(f'median  {median_n_nodes}')

    import matplotlib.pyplot as plt
    plt.hist(n_nodes_l, color='gray', bins=25)
    plt.title('histogram of IMDB-B graphs\' node counts')
    plt.xlabel('num. nodes')
    plt.ylabel('num. graphs')
    plt.show()

    



















## OLD: import graphs from txt files

# def import_IMDBB_graphs(path: str = './IMDB-BINARY',
#                         return_df: bool = True):
#     """
#     This imports the chrsmrrs text-files version of the
#     IMDB-BINARY dataset WHICH IS INCORRECT (1/3 graphs are disconnected...)
#     https://chrsmrrs.github.io/datasets/docs/datasets/
#     """
    
#     with open(f'{path}/IMDB-BINARY_graph_indicator.txt', 'r') as f:
#         graph_idxs = f.readlines()
#     with open(f'{path}/IMDB-BINARY_A.txt', 'r') as f:
#         edges = f.readlines()
#     with open(f'{path}/IMDB-BINARY_graph_labels.txt', 'r') as f:
#         graph_labels = f.readlines()
#         graph_labels = [int(l.strip('\n')) for l in graph_labels]

#     graph_i = 0
#     g = None
#     graphs_l = []
#     edges_l = []
    
#     for i, gidx_line in enumerate(graph_idxs):

#         # convert str graph index to int
#         gidx = int(gidx_line.strip('\n'))
        
#         # seeing a new gidx:
#         if gidx != graph_i:
    
#             # bank the current graph
#             if g is not None:
#                 g.add_edges_from(edges_l)
#                 graphs_l.append(g)
    
#             # start a new graph
#             graph_i = gidx
#             edges_l = []
#             g = nx.Graph()
#             edges_l = []
    
#         # for each line in graph_idxs
#         edge_s = edges[i]
#         edge_tup = tuple([int(x) for x in edge_s.split(', ')])
#         edges_l.append(edge_tup)
    
#     # bank the last graph
#     g.add_edges_from(edges_l)
#     graphs_l.append(g)

#     # put graphs and their labels in a df,
#     # else return tuple of graphs labels lists
#     if return_df:
#         import pandas as pd
#         graphs_df = pd.DataFrame({
#             'label': graph_labels,
#             'graph': graphs_l
#         })
#         return graphs_df
#     else:
#         return (graphs_l, graph_labels)
        

