import dataclasses
import enum
import math

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


class LeTETimeEncoder(nn.Module):
    """Learnable Transformation-based Generalized Time Encoder (LeTE).

    Implements the Combined LeTE version from "Rethinking Time Encoding via Learnable Transformation Functions"
    (Chen et al., arXiv:2505.00887).
    Uses a mix of Fourier-based and Spline-based (Tanh + MLP) learnable non-linear transformations.
    """

    def __init__(
        self,
        output_dim: int,
        p_fourier: float = 0.5,
        fourier_k: int = 5,
        spline_hidden_dim: int = 16,
        use_layernorm: bool = True,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.p_fourier = p_fourier
        self.fourier_k = fourier_k
        self.spline_hidden_dim = spline_hidden_dim
        self.use_layernorm = use_layernorm

        self.num_fourier_dims = math.floor(self.p_fourier * self.output_dim)
        self.num_spline_dims = self.output_dim - self.num_fourier_dims

        # Fourier parameters
        if self.num_fourier_dims > 0:
            self.omega_fourier = nn.Parameter(torch.empty(self.num_fourier_dims))
            self.phi_fourier = nn.Parameter(torch.empty(self.num_fourier_dims))
            self.a0_fourier = nn.Parameter(torch.empty(self.num_fourier_dims))
            if self.fourier_k > 0:
                self.ak_fourier = nn.Parameter(torch.empty(self.num_fourier_dims, self.fourier_k))
                self.bk_fourier = nn.Parameter(torch.empty(self.num_fourier_dims, self.fourier_k))
            self.register_buffer('k_vals_fourier', torch.arange(1, self.fourier_k + 1).float() if self.fourier_k > 0 else torch.empty(0))


        # Spline parameters
        if self.num_spline_dims > 0:
            self.omega_spline = nn.Parameter(torch.empty(self.num_spline_dims))
            self.phi_spline = nn.Parameter(torch.empty(self.num_spline_dims))
            # MLP: Linear(1, H) -> ReLU -> Linear(H, 1) applied element-wise
            self.spline_mlp_l1 = nn.Linear(1, self.spline_hidden_dim)
            self.spline_mlp_l2 = nn.Linear(self.spline_hidden_dim, 1)

        # Scaling factors
        self.scaling_factors_s = nn.Parameter(torch.empty(self.output_dim))

        # LayerNorm
        if self.use_layernorm and self.output_dim > 0:
            self.layer_norm = nn.LayerNorm(self.output_dim)

        self.reset_parameters()

    def reset_parameters(self):
        if self.num_fourier_dims > 0:
            nn.init.uniform_(self.omega_fourier, -0.05, 0.05)
            nn.init.zeros_(self.phi_fourier)
            nn.init.uniform_(self.a0_fourier, -0.05, 0.05)
            if self.fourier_k > 0:
                nn.init.uniform_(self.ak_fourier, -1.0 / math.sqrt(self.num_fourier_dims) if self.num_fourier_dims > 0 else 0.05, 1.0 / math.sqrt(self.num_fourier_dims) if self.num_fourier_dims > 0 else 0.05)
                nn.init.uniform_(self.bk_fourier, -1.0 / math.sqrt(self.num_fourier_dims) if self.num_fourier_dims > 0 else 0.05, 1.0 / math.sqrt(self.num_fourier_dims) if self.num_fourier_dims > 0 else 0.05)


        if self.num_spline_dims > 0:
            nn.init.uniform_(self.omega_spline, -0.05, 0.05)
            nn.init.zeros_(self.phi_spline)
            # MLP layers are initialized by default by PyTorch, which is usually good enough.

        nn.init.ones_(self.scaling_factors_s)


    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # Input t shape: (SeqLen, Batch, 1)
        # Output shape: (SeqLen, Batch, output_dim)


        if self.output_dim == 0:
            print("[LeTETimeEncoder] Output dim is 0, returning empty tensor.")
            return torch.empty((*t.shape[:-1], 0), device=t.device, dtype=t.dtype)

        t_flat = t.squeeze(-1)  # Shape (SeqLen, Batch)

        outputs = []

        # Fourier Part
        if self.num_fourier_dims > 0:
            # x_fourier = omega_f * t_flat.unsqueeze(-1) + phi_f
            # Unsqueeze t_flat for broadcasting with (num_fourier_dims)
            # omega_fourier and phi_fourier are (num_fourier_dims)
            # t_flat is (S, B) -> t_flat.unsqueeze(-1) is (S, B, 1)
            # x_fourier should be (S, B, num_fourier_dims)
            x_fourier = self.omega_fourier * t_flat.unsqueeze(2) + self.phi_fourier
            
            fourier_embedding = self.a0_fourier.unsqueeze(0).unsqueeze(0) # (1,1,num_fourier_dims)
            
            if self.fourier_k > 0:
                # k_vals_fourier: (K)
                # x_fourier: (S, B, num_fourier_dims)
                # ak_fourier, bk_fourier: (num_fourier_dims, K)
                
                # k_vals_fourier view: (1,1,1,K)
                k_view = self.k_vals_fourier.view(1, 1, 1, -1)
                # x_fourier view for broadcasting with k: (S,B,num_fourier_dims,1)
                x_fourier_expanded_k = x_fourier.unsqueeze(-1)

                cos_terms = torch.cos(k_view * x_fourier_expanded_k)  # (S,B,num_fourier_dims,K)
                sin_terms = torch.sin(k_view * x_fourier_expanded_k)  # (S,B,num_fourier_dims,K)
                
                # ak_fourier (num_fourier_dims, K) -> (1,1,num_fourier_dims,K)
                # bk_fourier (num_fourier_dims, K) -> (1,1,num_fourier_dims,K)
                ak_exp = self.ak_fourier.unsqueeze(0).unsqueeze(0)
                bk_exp = self.bk_fourier.unsqueeze(0).unsqueeze(0)
                
                fourier_series_sum = torch.sum(ak_exp * cos_terms + bk_exp * sin_terms, dim=-1) # (S,B,num_fourier_dims)
                fourier_embedding = fourier_embedding + fourier_series_sum
            
            outputs.append(fourier_embedding)

        # Spline Part (Tanh + MLP)
        if self.num_spline_dims > 0:
            # x_spline = omega_s * t_flat.unsqueeze(-1) + phi_s
            # omega_spline and phi_spline are (num_spline_dims)
            # x_spline should be (S, B, num_spline_dims)
            x_spline = self.omega_spline * t_flat.unsqueeze(2) + self.phi_spline

            tanh_base = torch.tanh(x_spline) # (S, B, num_spline_dims)

            # MLP part - applied element-wise for each of num_spline_dims
            # Reshape x_spline to (S * B * num_spline_dims, 1)
            s_dim, b_dim, n_spl_dim = x_spline.shape
            x_spline_reshaped = x_spline.reshape(-1, 1) # (S*B*num_spline_dims, 1)
            
            mlp_hidden = torch.relu(self.spline_mlp_l1(x_spline_reshaped)) # (S*B*num_spline_dims, spline_hidden_dim)
            mlp_out_reshaped = self.spline_mlp_l2(mlp_hidden) # (S*B*num_spline_dims, 1)
            
            spline_mlp_contribution = mlp_out_reshaped.reshape(s_dim, b_dim, n_spl_dim) # (S, B, num_spline_dims)
            
            spline_embedding = tanh_base + spline_mlp_contribution
            outputs.append(spline_embedding)

        if not outputs:
             # Should not happen if self.output_dim > 0
            print("[LeTETimeEncoder] No outputs from Fourier or Spline parts, returning empty tensor.")

        combined_output = torch.cat(outputs, dim=-1) # (SeqLen, Batch, output_dim)

        # LayerNorm and Scaling
        if self.use_layernorm and self.output_dim > 0:
            norm_output = self.layer_norm(combined_output)
        else:
            norm_output = combined_output
        
        final_output = self.scaling_factors_s * norm_output # Element-wise multiplication

        return final_output


class TripletEncoder(nn.Module, Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder.

    Copied from: https://github.com/pytorch/examples/blob/main/word_language_model/model.py
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        # Define Triplet Embedders
        self.date_embedder = LeTETimeEncoder(
            output_dim=cfg.token_dim,
            p_fourier=cfg.lete_p_fourier,
            fourier_k=cfg.lete_fourier_k,
            spline_hidden_dim=cfg.lete_spline_hidden_dim,
            use_layernorm=cfg.lete_use_layernorm
        )
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
        # Embed times and mask static value times
        time_emb = self.embed_func(self.date_embedder, time_delta_days) * ~static_mask.unsqueeze(dim=1)
        # Embed codes
        code_emb = self.code_embedder.forward(code).permute(0, 2, 1)
        # Embed numerical values and mask nan values
        val_emb = self.embed_func(self.numeric_value_embedder, numeric_value) * numeric_value_mask.unsqueeze(
            dim=1
        )

        # Sum the (time, code, value) triplets and
        embedding = time_emb + code_emb + val_emb

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
