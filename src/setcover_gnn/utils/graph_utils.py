import torch
import networkx as nx
import numpy as np
from typing import Union
import dgl


def get_qubo_graph(q_mat: Union[torch.Tensor, np.ndarray]) -> nx.Graph:
    """
    Convert QUBO matrix to NetworkX graph.
    
    Args:
        q_mat: QUBO matrix (square)
        
    Returns:
        NetworkX graph where edges represent non-zero QUBO entries
    """
    assert len(q_mat.shape) == 2 and q_mat.shape[0] == q_mat.shape[1], \
        "Matrix must be square"
    
    if isinstance(q_mat, torch.Tensor):
        q_mat = q_mat.detach().cpu().numpy()
    else:
        q_mat = np.array(q_mat)
    
    binary_mat = (q_mat != 0).astype(int)
    G = nx.from_numpy_array(binary_mat)
    return G


def pagerank_features(nx_graph: nx.Graph, feature_dim: int = 10) -> torch.Tensor:
    """
    Compute pagerank features for nodes.
    
    Args:
        nx_graph: NetworkX graph
        feature_dim: Dimension of output features
        
    Returns:
        Tensor of pagerank features
    """
    features = torch.zeros((nx_graph.number_of_nodes(), feature_dim))
    pagerank = nx.pagerank(nx.Graph(nx_graph))
    
    for k, v in pagerank.items():
        features[k, :] = v
    
    return features


def create_dgl_graph_from_qubo(qubo_matrix: torch.Tensor) -> dgl.DGLGraph:
    """Create DGL graph from QUBO matrix."""
    nx_graph = get_qubo_graph(qubo_matrix)
    dgl_graph = dgl.from_networkx(nx_graph=nx_graph)
    return dgl_graph