import copy
import os
# import dgl
import torch
import random
import numpy as np
import scanpy as sc
import scipy.sparse as sp
from easydict import EasyDict
from munkres import Munkres
from sklearn import metrics
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, accuracy_score
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics.cluster import *
from scipy.optimize import linear_sum_assignment as linear_assignment


class CellDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X)
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


def loader_construction(data_path):
    data = sc.read_h5ad(data_path)
    X_all = data.X
    y_all = data.obs.values[:,0]
    input_dim = X_all.shape[1]

    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=1)
    train_set = CellDataset(X_train, y_train)
    test_set = CellDataset(X_test, y_test)

    train_loader = DataLoader(dataset=train_set, batch_size=128, shuffle=True, num_workers=0)
    test_loader = DataLoader(dataset=test_set, batch_size=128, shuffle=False, num_workers=0)
    return train_loader, test_loader, input_dim


def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def evaluate(y_true, y_pred, num_classes):
    nmi = normalized_mutual_info_score(y_true, y_pred, average_method='arithmetic')
    ari = adjusted_rand_score(y_true, y_pred)
    homo = homogeneity_score(y_true, y_pred)
    comp = completeness_score(y_true, y_pred)
    new_predict = get_y_preds(y_true, y_pred, num_classes)
    acc= accuracy_score(y_true, new_predict)
    f1 = f1_score(y_true, new_predict, average='macro')
    return acc, f1, nmi, ari, homo, comp


def calculate_cost_matrix(C, n_clusters):
    cost_matrix = np.zeros((n_clusters, n_clusters))
    # cost_matrix[i,j] will be the cost of assigning cluster i to label j
    for j in range(n_clusters):
        s = np.sum(C[:, j])  # number of examples in cluster i
        for i in range(n_clusters):
            t = C[i, j]
            cost_matrix[j, i] = s - t
    return cost_matrix


def get_cluster_labels_from_indices(indices):
    n_clusters = len(indices)
    cluster_labels = np.zeros(n_clusters)
    for i in range(n_clusters):
        cluster_labels[i] = indices[i][1]
    return cluster_labels


def get_y_preds(y_true, cluster_assignments, n_clusters):
    confusion_matrix = metrics.confusion_matrix(np.array(y_true), np.array(cluster_assignments), labels=None)
    # compute accuracy based on optimal 1:1 assignment of clusters to labels
    cost_matrix = calculate_cost_matrix(confusion_matrix, n_clusters)
    indices = Munkres().compute(cost_matrix)
    kmeans_to_true_cluster_labels = get_cluster_labels_from_indices(indices)

    if np.min(cluster_assignments) != 0:
        cluster_assignments = cluster_assignments - np.min(cluster_assignments)
    y_pred = kmeans_to_true_cluster_labels[cluster_assignments]
    return y_pred


def aug_feature_shuffle(input_feat):
    """
    shuffle the features for fake samples.
    args:
        input_feat: input features
    returns:
        aug_input_feat: augmented features
    """
    fake_input_feat = input_feat[np.random.permutation(input_feat.shape[0]), :]
    return fake_input_feat


def random_gaussian_noise(cell_profiles, p=0.8):
    new_cell_profiles = cell_profiles.clone()
    num_samples, gene_num = cell_profiles.shape
    noise = torch.normal(0, 0.5, (num_samples, gene_num), device=cell_profiles.device)
    mask = torch.rand(num_samples, gene_num, device=cell_profiles.device) < p
    new_cell_profiles += noise * mask
    return new_cell_profiles


def re_features(features, h, hops, decimal_index, corrupt=False):
    sorted_indices = torch.argsort(decimal_index)
    sorted_features = features[sorted_indices]
    _, count = torch.unique(decimal_index, return_counts=True)
    # sorted_binary_index = torch.sign(h)[sorted_indices]
    # cos = F.cosine_similarity(h[sorted_indices], sorted_binary_index, dim=-1)
    # sorted_features = torch.cat([sorted_features, (sorted_indices/len(count)).unsqueeze(-1), cos.unsqueeze(-1)], dim=-1)

    if hops > min(count):
        hops = min(count)
    nodes_features = torch.empty(sorted_features.shape[0], 1, hops + 1, sorted_features.shape[1], device=features.device)
    nodes_features[:, 0, 0, :] = sorted_features
    if corrupt:
        sorted_aug_features = random_gaussian_noise(sorted_features)
        aug_nodes_features = torch.empty(sorted_aug_features.shape[0], 1, hops + 1, sorted_aug_features.shape[1], device=features.device)
        aug_nodes_features[:, 0, 0, :] = sorted_aug_features[sorted_indices]
    start = 0

    for i in range(count.shape[0]):
        sampled_indices = torch.stack([torch.randperm(count[i], device=features.device)[:hops] for _ in range(count[i])], dim=0)
        nodes_features[start:start+count[i], 0, 1:hops+1, :] = sorted_features[start+sampled_indices, :]
        if corrupt:
            aug_nodes_features[start:start+count[i], 0, 1:hops+1, :] = sorted_aug_features[start+sampled_indices, :]
        start += count[i]

    reverse_indices = torch.argsort(sorted_indices)
    if corrupt:
        return nodes_features[reverse_indices].squeeze(), aug_nodes_features[reverse_indices].squeeze()
    else:
        return nodes_features[reverse_indices].squeeze(), None


def get_index_mapping(decimal_index):
    unique_indices = torch.unique(decimal_index).to(torch.int64)
    new_indices = max(unique_indices).item() + 1
    index_mapping = {idx.item(): np.where(decimal_index == idx)[0] for idx in unique_indices}

    # if reverse_indexes is not None:
    #     index_mapping = {idx.item(): reverse_indexes[(decimal_index == idx).nonzero(as_tuple=True)[0]] for idx in unique_indices}
    # index_mapping = EasyDict(index_mapping)
            
    return index_mapping

def hop2token(features, hops, k, pe_dim=0):
    adj, graph = knn_graph(features, k)
    features = torch.from_numpy(features)
    if pe_dim > 0:
        lpe = laplacian_positional_encoding(graph, pe_dim)
        features = torch.cat((features, lpe), dim=1)
    aug_features = aug_feature_shuffle(features)
    nodes_features = torch.empty(features.shape[0], 1, hops + 1, features.shape[1])
    aug_nodes_features = torch.empty(aug_features.shape[0], 1, hops + 1, aug_features.shape[1])

    # for i in range(features.shape[0]):
    #     nodes_features[i, 0, 0, :] = features[i]
    nodes_features[:, 0, 0, :] = features
    aug_nodes_features[:, 0, 0, :] = aug_features

    x = features + torch.zeros_like(features)
    aug_x = aug_features + torch.zeros_like(aug_features)

    for i in range(hops):
        x = torch.matmul(adj, x)
        aug_x = torch.matmul(adj, aug_x)
        nodes_features[:, 0, i + 1, :] = x[:]
        aug_nodes_features[:, 0, i + 1, :] = aug_x

    re_features = torch.cat([nodes_features, aug_nodes_features], dim=1)
    return re_features, adj, graph


def knn_graph(features, k):
    # features (N, d)
    # Create a kNN graph according to the input features
    N, d = features.shape
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(features)
    distances, indices = nbrs.kneighbors(features)

    # Prepare data for sparse adjacency matrix
    row_indices = np.repeat(np.arange(N), k)
    col_indices = indices.flatten()
    values = np.ones(len(row_indices), dtype=np.float64)

    # Create a sparse adjacency matrix using scipy
    adj = sp.coo_matrix((values, (row_indices, col_indices)), shape=(N, N))
    ####################################

    # Compute the degree matrix
    degree = np.array(adj.sum(1)).flatten()
    degree_inv_sqrt = np.power(degree, -0.5)
    degree_inv_sqrt[np.isinf(degree_inv_sqrt)] = 0.0

    # Create a diagonal matrix for D^{-1/2}
    D_inv_sqrt = sp.diags(degree_inv_sqrt)

    # Compute the normalized adjacency matrix
    norm_adj = D_inv_sqrt @ adj @ D_inv_sqrt
    graph = dgl.from_scipy(norm_adj)

    # Convert the normalized adjacency matrix to a torch sparse tensor
    norm_adj = sparse_mx_to_torch_sparse_tensor(norm_adj)

    return norm_adj, graph


def laplacian_positional_encoding(g, pos_enc_dim):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """

    # Laplacian

    #adjacency_matrix(transpose, scipy_fmt="csr")
    # A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    A = g.adj_external(scipy_fmt='csr')
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    # Eigenvectors with scipy
    #EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR')
    EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR', tol=1e-2) # for 40 PEs
    EigVec = EigVec[:, EigVal.argsort()] # increasing order
    lap_pos_enc = torch.from_numpy(EigVec[:,1:pos_enc_dim+1]).float()

    return lap_pos_enc


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float64)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.LongTensor(indices, values, shape)


def kmeans(x, n_clusters, centers=None):
    if centers is None:
        if type(x) == torch.Tensor:
            x = x.cpu().detach().numpy()
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(x)
        centers = kmeans.cluster_centers_
        y_pred = kmeans.labels_
    else:
        # x = torch.from_numpy(x)
        distance = dis_fun(x, centers)
        y_pred = torch.argmin(distance, dim=-1).cpu().detach().numpy()
    if type(y_pred) == torch.Tensor:
        y_pred = y_pred.cpu().detach().numpy()
    if type(centers) == torch.Tensor:
        centers = centers.cpu().detach().numpy()
    return y_pred, centers


def dis_fun(x, c):
    xx = (x * x).sum(-1).reshape(-1, 1).repeat(1, c.shape[0])
    cc = (c * c).sum(-1).reshape(1, -1).repeat(x.shape[0], 1)
    xx_cc = xx + cc
    xc = x @ c.T
    distance = xx_cc - 2 * xc
    return distance


def soft_assign(z, centers):
    q1 = 1.0 / (1 + torch.cdist(z, centers, p=2) ** 2)  # N x K 的距离矩阵
    q = q1 / q1.sum(dim=1).unsqueeze(1)

    p = q ** 2 / q.sum(0)
    return q, (p.t() / p.sum(1)).t()
