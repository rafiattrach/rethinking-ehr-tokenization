import dataclasses
import enum
import warnings
import sys
from pathlib import Path

import torch
from omegaconf import DictConfig
from torch import nn
from loguru import logger

# --- Dynamic Import Logic (No changes needed here) ---
_TIME2VEC_IMPORTED = False
try:
    project_root = Path(__file__).resolve().parents[4]
    time2vec_repo_path = project_root / 'time2vec'
    if not time2vec_repo_path.is_dir(): raise ImportError(f"Dir not found: {time2vec_repo_path}")
    if str(time2vec_repo_path) not in sys.path:
        sys.path.insert(0, str(time2vec_repo_path))
        logger.debug(f"Added '{time2vec_repo_path}' to sys.path")
    from time2vec.torch.time2vec_torch_keras_init import Time2Vec as Time2VecTorch
    logger.info("Successfully imported Time2VecTorch from adjacent repository.")
    _TIME2VEC_IMPORTED = True
except Exception as e:
    logger.error(f"Failed to import Time2Vec: {e}")
    raise ImportError("Could not import Time2Vec.") from e
# --- End dynamic import ---

from meds_torch.input_encoder import INPUT_ENCODER_MASK_KEY, INPUT_ENCODER_TOKENS_KEY
from meds_torch.utils.module_class import Module

# --- ModelOutput, Triplet Enum (No Changes) ---
@dataclasses.dataclass
class ModelOutput:
    rep: torch.Tensor
    hidden_states: torch.Tensor = None

class Triplet(enum.Enum):
    DATE = "date"
    VALUE = "value"
    VARIABLE = "variable"

# --- CVE class (No Changes) ---
class CVE(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        if input_dim <= 0: raise ValueError(f"CVE input_dim must be positive, got {input_dim}")
        self.layer = nn.Linear(input_dim, output_dim)
        logger.debug(f"Initialized CVE: Linear({input_dim}, {output_dim})")
    def forward(self, x): return self.layer(x)

# --- Corrected TripletEncoder for RQ1 ---
class TripletEncoder(nn.Module, Module):
    """Encodes (time_delta, code, numeric_value) triplets.

    MODIFIED FOR RQ1: Uses Time2Vec (imported) for time_delta, ensuring device compatibility.
    """
    def __init__(self, cfg: DictConfig):
        super().__init__()
        if not _TIME2VEC_IMPORTED: raise RuntimeError("Time2Vec not imported.")
        self.cfg = cfg
        logger.info("RQ1: Initializing TripletEncoder using Time2Vec for time deltas.")

        # Validate config
        required_keys = ["time2vec_k", "token_dim", "vocab_size", "max_seq_len"]
        for key in required_keys:
            if key not in cfg: raise KeyError(f"Config missing key '{key}'.")
        self.time2vec_k = int(cfg.time2vec_k)
        self.token_dim = int(cfg.token_dim)
        self.vocab_size = int(cfg.vocab_size)
        self.max_seq_len = int(cfg.max_seq_len)
        if self.time2vec_k <= 0: raise ValueError("'time2vec_k' must be positive.")
        if self.token_dim <= 0: raise ValueError("'token_dim' must be positive.")

        # 1. Time Delta Encoding: Time2Vec + Projection
        # We still initialize it here, but need to manage device placement in forward
        self.date_embedder = Time2VecTorch(num_frequency=self.time2vec_k, num_vars=1)
        time2vec_output_dim = 1 + self.time2vec_k
        self.time_projection = nn.Linear(time2vec_output_dim, self.token_dim)
        logger.debug(f"Time Encoder: Time2Vec(k={self.time2vec_k}, out={time2vec_output_dim}) -> Linear(->{self.token_dim})")

        # 2. Code Encoding: Standard Embedding
        self.code_embedder = nn.Embedding(self.vocab_size, embedding_dim=self.token_dim, padding_idx=0)
        logger.debug(f"Code Encoder: Embedding({self.vocab_size}, {self.token_dim})")

        # 3. Numeric Value Encoding: CVE
        self.numeric_value_embedder = CVE(input_dim=1, output_dim=self.token_dim)
        logger.debug(f"Value Encoder: CVE(1, {self.token_dim})")

        # Flag to track if Time2Vec periodic params have been initialized *on the correct device*
        self._time2vec_device_initialized = False


    def get_embedding(self, batch):
        # Extract & Type Check
        try:
            static_mask = batch["static_mask"].bool()
            code = batch["code"].long()
            numeric_value = batch["numeric_value"].float()
            time_delta_days = batch["time_delta_days"].float()
            numeric_value_mask = batch["numeric_value_mask"].bool()
        except KeyError as e: raise KeyError(f"Missing key '{e}' in input batch.") from e
        except Exception as e: raise TypeError(f"Error casting batch tensors: {e}") from e

        B, S = code.shape
        current_device = time_delta_days.device # Get the device the batch is on

        # 1. Encode Time Delta using Time2Vec
        time_input = time_delta_days.unsqueeze(-1) # (B, S, 1)

        # --- Device Handling for Imported Time2Vec ---
        # Check if Time2Vec internal params need device sync or init
        # The internal periodic_weight might be None or on CPU initially
        if self.date_embedder.periodic_weight is None or \
           self.date_embedder.periodic_weight.device != current_device:
             # Call the internal init function if available OR manually move params
             if hasattr(self.date_embedder, 'initialize_periodic_parameters'):
                 logger.debug(f"Initializing Time2Vec periodic params on device {current_device}")
                 # This function in the imported code handles creation and device placement
                 self.date_embedder.initialize_periodic_parameters(time_input.shape)
                 # Explicitly ensure module parameters are moved (might be redundant if init does it)
                 self.date_embedder.to(current_device)
             else:
                 # Fallback: Manually move existing parameters if init function is missing
                 logger.warning("Time2Vec module missing 'initialize_periodic_parameters'. "
                                f"Attempting manual parameter move to device {current_device}.")
                 self.date_embedder.to(current_device) # Move the whole module + its params
        # --- End Device Handling ---

        time_vec = self.date_embedder(time_input)       # (B, S, 1+k) - Now should work on correct device
        time_proj = self.time_projection(time_vec)      # (B, S, D)
        time_emb = time_proj.permute(0, 2, 1)           # (B, D, S)
        time_emb = time_emb * (~static_mask.unsqueeze(1)) # Mask static

        # 2. Encode Code using Embedding (already on correct device via Lightning)
        code_emb = self.code_embedder(code).permute(0, 2, 1) # (B, D, S)

        # 3. Encode Numeric Value using CVE (already on correct device via Lightning)
        value_input = numeric_value.unsqueeze(-1)       # (B, S, 1)
        value_vec = self.numeric_value_embedder(value_input) # (B, S, D)
        value_permuted = value_vec.permute(0, 2, 1)     # (B, D, S)
        val_emb = value_permuted * numeric_value_mask.unsqueeze(1) # Mask missing

        # --- Summation ---
        embedding_sum = time_emb + code_emb + val_emb      # (B, D, S)

        # --- Checks ---
        if not torch.isfinite(embedding_sum).all():
             warnings.warn("Non-finite values detected in summed embedding! Replacing with zeros.")
             embedding_sum = torch.nan_to_num(embedding_sum, nan=0.0, posinf=0.0, neginf=0.0)

        # --- Final Transpose ---
        final_embedding = embedding_sum.transpose(1, 2) # (B, S, D)
        return final_embedding

    def forward(self, batch):
        embedding = self.get_embedding(batch)
        if 'mask' not in batch: raise KeyError("Batch must contain padding 'mask'.")
        batch[INPUT_ENCODER_TOKENS_KEY] = embedding
        batch[INPUT_ENCODER_MASK_KEY] = batch["mask"].bool() # Use padding mask
        return batch