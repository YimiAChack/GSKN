# GSKN: Theoretically Improving Graph Neural Networks via Anonymous Walk Graph Kernels

The repository implements Graph Convolutional Kernel Networks (GCKNs) described in the following paper
> GSKN: Theoretically Improving Graph Neural Networks via Anonymous Walk Graph Kernels (WWW 2021, Research Track, Full Paper)

For more details, please see our [Paper](https://arxiv.org/submit/3687651).

#### Installation
We strongly recommend users to use miniconda to install the following packages (link to pytorch)

```
python=3.6.2
numpy
scikit-learn=0.21
pytorch=1.3.1
torchvision=0.4.2
pandas
networkx
Cython
cyanure
```

All the above packages can be installed with `conda install` except `cyanure`, which can be installed with `pip install cyanure-mkl`.

CUDA Toolkit also needs to be downloaded with the same version as used in Pytorch. Then place it under the path `$PATH_TO_CUDA` and run `export CUDA_HOME=$PATH_TO_CUDA`.

Finally run `make`, and it may take few minutes to compile.


#### Data


Run `cd dataset; bash get_data.sh` to download and unzip datasets. We provide here 3 types of datasets: datasets without node attributes (IMDBBINARY, IMDBMULTI, COLLAB), datasets with discrete node attributes (MUTAG, PROTEINS, PTC) and datasets with continuous node attributes (BZR, COX2, PROTEINS_full). All the datasets can be downloaded and extracted from [this site](https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets).

#### run

```
export PYTHONPATH=$PWD:$PYTHONPATH
python main.py --dataset MUTAG  --sigma 1.5 --hidden_size 16  --aggregation --anonymous_walk_length 6 --anonymous_walks_per_node 30
```

#### Acknowledgments
Certain parts of this project are partially derived from [GCKN](https://github.com/claying/GCKN) and [GraphSTONE](https://github.com/YimiAChack/GraphSTONE).
