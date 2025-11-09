import dataclasses
import enum

import torch
from omegaconf import DictConfig
from torch import nn

from meds_torch.input_encoder import INPUT_ENCODER_MASK_KEY, INPUT_ENCODER_TOKENS_KEY
from meds_torch.utils.module_class import Module


@dataclasses.dataclass
class ModelOutput:
    rep: torch.Tensor
    hidden_states: torch.Tensor = None


class Triplet(enum.Enum):
    DATE = "date"
    VALUE = "value"
    VARIABLE = "variable"


def sequence_mask(lengths, maxlen, dtype=torch.bool):
    row_vector = torch.arange(0, maxlen, 1, device=lengths.device)
    matrix = torch.unsqueeze(lengths, dim=-1)
    mask = row_vector < matrix

    mask.type(dtype)
    return mask


class CVE(nn.Module):
    """Continuous Value Encoder (CVE) module.

    Assumes input is a single continuous value, and encodes it as an `output_dim` size embedding vector.
    """

    def __init__(self, cfg):
        super().__init__()
        self.layer = nn.Linear(1, cfg.token_dim)

    def forward(self, x):
        return self.layer(x)


class TripletEncoder(nn.Module, Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder.
    
    MODIFIED FOR RQ1 CODE-ONLY: Excludes both time and value embeddings from triplet representation 
    to test if code embeddings alone are sufficient for clinical prediction tasks.

    Copied from: https://github.com/pytorch/examples/blob/main/word_language_model/model.py
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        # Define Triplet Embedders
        # NOTE: date_embedder and numeric_value_embedder are kept for compatibility but won't be used in final embedding
        self.date_embedder = CVE(cfg)
        self.code_embedder = torch.nn.Embedding(cfg.vocab_size, embedding_dim=cfg.token_dim)
        self.numeric_value_embedder = CVE(cfg)

    def embed_func(self, embedder, x):
        out = embedder.forward(x[None, :].transpose(2, 0)).permute(1, 2, 0)
        return out

    def get_embedding(self, batch):
        static_mask = batch["static_mask"]
        code = batch["code"]
        numeric_value = batch["numeric_value"]
        time_delta_days = batch["time_delta_days"]
        numeric_value_mask = batch["numeric_value_mask"]
        
        # RQ1 CODE-ONLY: Comment out both time and value embeddings to test if codes alone are sufficient
        # time_emb = self.embed_func(self.date_embedder, time_delta_days) * ~static_mask.unsqueeze(dim=1)
        # val_emb = self.embed_func(self.numeric_value_embedder, numeric_value) * numeric_value_mask.unsqueeze(dim=1)
        
        # Embed codes only
        code_emb = self.code_embedder.forward(code).permute(0, 2, 1)

        # RQ1 CODE-ONLY: Use only code embeddings, excluding both time and value components
        embedding = code_emb

        assert embedding.isfinite().all(), "Embedding is not finite"
        if embedding.shape[-1] > self.cfg.max_seq_len:
            raise ValueError(
                f"Triplet embedding length {embedding.shape[-1]} "
                "is greater than max_seq_len {self.cfg.max_seq_len}"
            )
        return embedding.transpose(1, 2)

    def forward(self, batch):
        embedding = self.get_embedding(batch)
        batch[INPUT_ENCODER_MASK_KEY] = batch["mask"]
        batch[INPUT_ENCODER_TOKENS_KEY] = embedding
        return batch
