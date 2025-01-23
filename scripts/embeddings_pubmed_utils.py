import torch
import numpy as np
import random


def fix_all_seeds(seed=42):
    """Fix all seeds when working with pytorch models
    """
    # Set the random seed for PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  ## this one is new
    ## Set the seed for generating random numbers on all GPUs.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True) ## this one I don't use but don't remember why

    # Set the random seed for NumPy
    np.random.seed(seed)

    # Set the random seed
    random.seed(seed)



def mean_pooling(token_embeds, attention_mask):
    """Returns [AVG] token.
    Returns average embedding of the embeddings of all tokens of a corpus ([AVG]).

    Parameters
    ----------
    token_embeds : torch of shape (n_documents, 512, 768)
        First element of model_output contains all token embeddings (model_output[0])
    attention_mask : inputs["attention_mask"], inputs being the output of the tokenizer

    """
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeds.size()).float()
    )
    sum_embeddings = torch.sum(token_embeds * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def sep_pooling(token_embeds, attention_mask):
    """Returns [SEP] token
    Returns [SEP] token from all the embeddings of all tokens of a corpus.

    Parameters
    ----------
    token_embeds : torch of shape (n_documents, 512, 768)
        First element of model_output contains all token embeddings (model_output[0])
    attention_mask : inputs["attention_mask"], inputs being the output of the tokenizer

    """
    ix = attention_mask.sum(1) - 1
    ix0 = torch.arange(attention_mask.size(0))
    return token_embeds[ix0, ix, :]


@torch.no_grad()
def generate_embeddings_batches(abstracts, tokenizer, model, device):
    """Generate embeddings using BERT-based model.

    Parameters
    ----------
    abstracts : list, this has to be a list not sure if array works but pandas do not work
        Abstract texts.
    tokenizer : transformers.models.bert.tokenization_bert_fast.BertTokenizerFast
        Tokenizer.
    model : transformers.models.bert.modeling_bert.BertModel
        BERT-based model.
    device : str, {"cuda", "cpu"}
        "cuda" if torch.cuda.is_available() else "cpu".

    Returns
    -------
    embedding_cls : ndarray
        [CLS] tokens of the abstracts.
    embedding_sep : ndarray
        [SEP] tokens of the abstracts.
    embedding_av : ndarray
        Average of tokens of the abstracts.
    """
    # preprocess the input
    inputs = tokenizer(
        abstracts,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=512,
    ).to(device)

    with torch.no_grad():
        model.eval()
        out = model(**inputs)
        token_embeds = out[0]  # get the last hidden state
        av = mean_pooling(token_embeds, inputs["attention_mask"])
        sep = sep_pooling(token_embeds, inputs["attention_mask"])
        cls = token_embeds[:, 0, :]
        embedding_av = av.detach().cpu().numpy()
        embedding_sep = sep.detach().cpu().numpy()
        embedding_cls = cls.detach().cpu().numpy()

    return embedding_cls, embedding_sep, embedding_av