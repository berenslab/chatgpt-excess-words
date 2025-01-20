import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from pathlib import Path
import time
import os

from process_pubmed_utils import fix_all_seeds, generate_embeddings_batches

# define paths
variables_path = Path("../results/variables")
print(variables_path.resolve())
berenslab_data_path = Path("/gpfs01/berens/data/data/pubmed_processed")

# import data
df = pd.read_csv(berenslab_data_path / "pubmed_baseline_2025.zip")

# extract abstract texts from years 2022-2024
abstracts = df["AbstractText"][(df.Year >= 2022) & (df.Year <= 2024)].tolist()
print(f"There are {len(abstracts)} papers in 2022-2024")


# fix random seeds
fix_all_seeds()

# specify model
model_name = "PubMedBERT"
model_path = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"

# set up model
print("Model: ", model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Running on device: {}".format(device))
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)
print(model_path)

model.to(device)


# compute embeddings
start = time.time()
saving_path = Path("embeddings/2022-2024_papers")

loader = torch.utils.data.DataLoader(abstracts, batch_size=256, num_workers=0)

embedding_av = []
embedding_sep = []
embedding_cls = []

for i_batch, batch in enumerate(tqdm(loader)):
    embd_cls, embd_sep, embd_av = generate_embeddings_batches(
        batch, tokenizer, model, device
    )
    embedding_av.append(embd_av)
    embedding_cls.append(embd_cls)
    embedding_sep.append(embd_sep)

    if (i_batch % 200) == 0:
        np.save(berenslab_data_path / saving_path / "last_i_batch", i_batch)
        np.save(
            berenslab_data_path / saving_path / "embedding_av_interm",
            np.vstack(embedding_av),
        )
        np.save(
            berenslab_data_path / saving_path / "embedding_cls_interm",
            np.vstack(embedding_cls),
        )
        np.save(
            berenslab_data_path / saving_path / "embedding_sep_interm",
            np.vstack(embedding_sep),
        )


# save all
np.save(
    berenslab_data_path / saving_path / "embedding_av_all",
    np.vstack(embedding_av),
)
np.save(
    berenslab_data_path / saving_path / "embedding_cls_all",
    np.vstack(embedding_cls),
)
np.save(
    berenslab_data_path / saving_path / "embedding_sep_all",
    np.vstack(embedding_sep),
)

end = time.time()
runtime_total = end - start
print("Total runtime: ", runtime_total)
np.save(berenslab_data_path / saving_path / "runtime_total", runtime_total)