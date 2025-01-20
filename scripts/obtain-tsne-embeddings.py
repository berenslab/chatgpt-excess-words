import pandas as pd
import numpy as np
from openTSNE import TSNE, affinity
from pathlib import Path
import time
import humanize

# define paths
variables_path = Path("../results/variables")
berenslab_data_path = Path("/gpfs01/berens/data/data/pubmed_processed")

# import embeddings
saving_path = Path("embeddings/2022-2024_papers")
embedding_sep = np.load(
    berenslab_data_path / saving_path / "embedding_sep_all.npy",
)

## t-SNE
start = time.time()

A = affinity.Uniform(
    embedding_sep,
    verbose=True,
    random_state=42,
    k_neighbors=10,
)

tsne_sep = TSNE(
    verbose=True, initialization="pca", random_state=42
).fit(affinities=A)

end = time.time()
runtime_total = end - start
print("Total runtime: ", runtime_total)
print(humanize.precisedelta(runtime_total))

# save
np.save(
    variables_path / "tsne_sep",
    tsne_sep,
)