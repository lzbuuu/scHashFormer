# Hash-Driven Tokenization Enables Scalable Transformer Models for scRNA-seq Analysis
## Abstract
The code of *scHashFormer: enhancing efficiency and scalability of scRNA-seq data clustering via a hash-based Transformer*.
In this paper, we introduce a hash-driven tokenization mechanism, where we design a neural network to simulate the locally sensitive hashing function and train it by self-supervised learning to realize similar cells with the same hash codes. Extensive experiments demonstrate that our mechanism outperforms similarity-based methods in downstream tasks, such as cell-type identification, trajectory preservation, and gene differential expression analysis. In addition, it scales efficiently to large-scale datasets, delivering up to an order-of-magnitude speedup, avoiding out-of-memory failures, and providing a principled framework for enabling Transformer models in unsupervised scRNA-seq analysis.

## Architecture
The neural network architecture of scHashFormer. ![fram1 (1)](./scHashFormer.png)

## Requirements
Creating a conda environment named `env_name` and installing the requirements using pip 
```
$ conda create --name [env_name] python=3.11
$ conda activate [env_name] 
$ pip install -r requirements.txt
```

## Datasets and corresponding configs
Downloading the scRNA-seq data named `dataset_name` to the path `dataset_path` and modifying the `data_dir` in corresponding configuration file `configs\[dataset_name].yml` to `dataset_name`.

The specific data file can be downloaded from the following website:
- The Chen data downloaded from [GSE87544](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE87544),
- The Bach data downloaded from [GSE106273](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE106273),
- The mouse retina cell atlas data (MRCA) downloaded from [UCSC Cell Browser](https://cells.ucsc.edu/muscle-cell-atlas/),
- The human retina - retinal ganglion cells data (HRCA) downloaded from [Single Cell Portal](https://singlecell.broadinstitute.org/single_cell/study/SCP2808/hrca-snrna-seq-of-the-human-retina-retinal-ganglion-cells),
- The human fetal atlas data (Fetal-Atlas) downloaded from [GSE156793](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE156793),
- The Human PBMC data downloaded from [Kaggle](https://www.kaggle.com/competitions/open-problems-single-cell-perturbations/data),
- The cardiovascular cell of the healthy Wistar rat data (Ratmap) downloaded from [Single Cell Portal](https://singlecell.broadinstitute.org/single_cell/study/SCP2828/transcriptional-profile-of-the-rat-cardiovascular-system-at-single-cell-resolution),
- The astrocytes cell downloaded from [Neuroscience Multi-omic Data Archive](https://data.nemoarchive.org/biccn/grant/u01_feng/feng/transcriptome/sncell/10x_v3.1/),
- The arabidopsis cell downloaded from [GSE290214](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE290214).

## Usage
For the Bach dataset, run
```
$ python main.py --dataset Bach
```

## Citation
