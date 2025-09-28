import argparse
import data
from config import get_config
import numpy as np
import scanpy as sc
from scipy import sparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default='./data')
    parser.add_argument("--dataset", type=str, default='HRCA')
    parser.add_argument("--highly_genes", type=int, default=2048)
    args = parser.parse_args()
    return args


def preprocess(adata):
    if isinstance(adata.X, sparse.csr_matrix) or isinstance(adata.X, sparse.csc_matrix):
        adata.X = adata.X.toarray()
    raw = adata.X.copy()
    
    sc.pp.normalize_total(adata, target_sum=1e4)
    sf = np.array((raw.sum(axis=1) / 1e4).tolist()).reshape(-1, 1)
    adata.obs['sf'] = sf

    sc.pp.log1p(adata)

    if adata.shape[1] < 5000:
        sc.pp.highly_variable_genes(adata, n_top_genes=3000)
    else:
        sc.pp.highly_variable_genes(adata)
    hvg_index = adata.var["highly_variable"].values
    raw = raw[:, hvg_index]
    adata = adata[:, hvg_index]
    adata.raw = raw
    sc.pp.scale(adata, max_value=10)
    x = adata.X

    return adata


if __name__ == "__main__":
    args = parse_args()
    args.dataset = 'Ratmap'

    args.path = '/home/user/data_zyx/datasets/single-cell/Ratmap-scp/ratmap_scp.h5ad'
    print(f"Loading dataset {args.dataset} from {args.path}...")
    if '.h5ad' in args.path:
        adata = data.load_h5ad(args.path)
    else:
        X, Y = data.load_h5(args.path)
        X = np.ceil(X).astype(np.float64)
        adata = sc.AnnData(X)
        adata.obs['Group'] = Y
    shape = adata.X.shape
    num_classes = max(adata.obs['Group'].values) + 1
    print(f"Loaded dataset {args.dataset} with {num_classes} classes, {shape[0]} cells and {shape[1]} genes.")

    print(f"Preprocessing dataset {args.dataset}...")
    # adata = data.normalize(adata, copy=True, highly_genes=args.highly_genes, size_factors=True, normalize_input=True, logtrans_input=True)
    adata = preprocess(adata)
    shape = adata.X.shape
    print(f"Preprocessed dataset {args.dataset} with {num_classes} classes, {shape[0]} cells and {shape[1]} genes.")

    if '.h5ad' in args.path:
        save_path = args.path.replace('.h5ad', f'_preprocessed_{args.highly_genes}.h5ad')
    else:
        save_path = args.path.replace('.h5', f'_preprocessed_{args.highly_genes}.h5ad')

    data.save_h5ad(adata, save_path=save_path)
    print(f"Saved processed dataset {args.dataset} to {save_path}.")
