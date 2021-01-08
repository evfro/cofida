# CoFiDa
This is a collection of fast in-memory pytorch dataloaders designed specifically for **co**llaborative **fi**ltering **da**tasets (hence the name of the project). The main goal for these dataloaders is to minimize waiting time on IO operations when training neural collaborative filtering models.

## Key features:
- vectorized numpy-style batch indexing, which avoids element-wise operations,
- reduced IO between CPU and GPU, which otherwise can be unnecesarily costly for collaborative filtering datasets comparing to actual computation time,
- support for different sampling scenarios (standard sample batch, user-wise batch, BPR sampling) that enables fast training of both NCF-like and autoencoder-based recommenders,
 - designed to provide a simple interface for data manipulation identical to standard PyTorch dataloaders.

All operations, except negative sampling, are performed directly on GPU, which is suitable for large yet extremely sparse datasets. Implementation is motivated by the lack of support of in-memory data manipulation in PyTorch standard library and is inspired by discussion at https://github.com/pytorch/pytorch/issues/21645.

## Dependencies
- numpy
- scipy
- pytorch
- polara