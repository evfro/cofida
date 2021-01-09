# CoFiDa
This is a collection of fast in-memory pytorch dataloaders designed specifically for **co**llaborative **fi**ltering **da**tasets (hence the name of the project). The main goal for these dataloaders is to minimize waiting time on IO operations when training neural collaborative filtering models.

## Key features:
- vectorized numpy-style batch indexing, which avoids element-wise operations,
- reduced IO between CPU and GPU, which otherwise can be unnecesarily costly for collaborative filtering datasets comparing to actual computation time,
- support for different sampling scenarios (standard sample batch, user-wise batch, BPR sampling) that enables fast training of both NCF-like and autoencoder-based recommenders,
 - designed to provide a simple interface for data manipulation identical to standard PyTorch dataloaders.

All operations, except negative sampling, are performed directly on GPU, which is suitable for large yet extremely sparse datasets. Implementation is motivated by the lack of support of in-memory data manipulation in PyTorch standard library and is inspired by discussion at https://github.com/pytorch/pytorch/issues/21645.

## Usage example:
```python
import numpy as np
from cofida.datasets import observations_loader, UserBatchDataset

# standard coordinate format for user-item interactions (can also have 3rd column for rating),
# data can also be in scipy's sparse CSR format
observations = np.array([
    [0, 0, 1, 2, 3], [0, 2, 2, 3, 1]
]).T

# initializing similarly to standard pytorch dataloaders,
# user-based batch is used here
data_loader = observations_loader(observations, batch_size=2, data_factory=UserBatchDataset)

# main learning part
for batch in data_loader: # this is where most of the waiting time is shaved off
    do_some_learning()
```

## Dependencies
- numpy
- scipy
- pytorch
- [polara](https://github.com/evfro/polara)
