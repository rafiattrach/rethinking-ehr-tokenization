"""
Flexible TextCode Encoder for RQ3 Experiments
Supports all 6 variants through parameters:
- mapping_type: "original" vs "enhanced"
- model_type: "tinybert", "bigbert", "qwen3" 
- freeze_model: true/false
"""

import dataclasses
import enum
import time

import polars as pl
import torch
from mixins import TimeableMixin
from omegaconf import DictConfig
from torch import nn
from transformers import AutoModel, AutoTokenizer
import pandas as pd
from typing import Dict, List, Tuple, Optional
import os

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


def fast_unique_with_inverse(x):
    """Efficiently computes unique elements and their inverse mapping for a 2D tensor.

    The function returns a tuple containing:
    - unique: tensor of unique values in sorted order
    - inverse: tensor of same shape as input, where each element is replaced by
              its index in the unique tensor

    Args:
        x (torch.Tensor): 2D input tensor with values in range [0, 10]

    Returns:
        tuple: (unique values tensor, inverse mapping tensor)

    Example:
        >>> x = torch.tensor([[0, 1, 0],
        ...                   [2, 1, 0]], device='cpu')
        >>> unique, inverse = fast_unique_with_inverse(x)
        >>> print(unique)
        tensor([0, 1, 2])
        >>> print(inverse)
        tensor([[0, 1, 0],
                [2, 1, 0]])

        >>> # Test with repeated values
        >>> x = torch.tensor([[5, 5, 5],
        ...                   [3, 3, 5]], device='cpu')
        >>> unique, inverse = fast_unique_with_inverse(x)
        >>> print(unique)
        tensor([3, 5])
        >>> print(inverse)
        tensor([[1, 1, 1],
                [0, 0, 1]])

        >>> # Test with all possible values
        >>> x = torch.tensor([[0, 10, 5],
        ...                   [7, 3, 1]], device='cpu')
        >>> unique, inverse = fast_unique_with_inverse(x)
        >>> print(unique)
        tensor([ 0,  1,  3,  5,  7, 10])
        >>> print(inverse)
        tensor([[0, 5, 3],
                [4, 2, 1]])
    """
    # Pre-allocate an empty tensor spanning the range of possible values
    B = torch.zeros(x.max().item() + 1, device=x.device, dtype=torch.int64)

    # First mark which positions have values (with 1s)
    B.scatter_(0, x.flatten(), torch.ones_like(x.flatten()))

    # Get unique values
    unique = torch.nonzero(B).flatten()

    # Create inverse mapping
    inverse = torch.zeros_like(x)
    for i, val in enumerate(unique):
        inverse[x == val] = i

    return unique, inverse


class FlexibleTextCodeEmbedder(nn.Module, Module, TimeableMixin):
    """
    Flexible TextCode Embedder that supports all RQ3 experiment variants
    """
    
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        
        # Initialize minimal stats tracking
        self.stats = {
            'total_forward_passes': 0,
            'total_duration_ms': 0.0,
            'total_unique_codes': 0,
            'cache_hits': 0,
            'bert_inferences': 0
        }
        
        # 🎯 VERIFICATION PRINTS
        print(f"🎯 VERIFICATION: Loading {self.__class__.__name__}")
        print(f"🎯 VERIFICATION: Model = {cfg.code_embedder}")
        print(f"🎯 VERIFICATION: Mapping = {cfg.code_metadata_fp}")
        print(f"🎯 VERIFICATION: Frozen = {getattr(cfg, 'freeze_model', False)}")
        print(f"🎯 VERIFICATION: Tokenizer config = {cfg.tokenizer_config}")
        
        # Check if we should use precomputed cache (for frozen models)
        self.use_cache = getattr(cfg, 'freeze_model', False)
        
        if self.use_cache:
            # For frozen models, try to load precomputed embeddings
            cache_path = self._get_cache_path()
            if cache_path.exists():
                print(f"🎯 VERIFICATION: Loading precomputed embeddings from {cache_path}")
                self._load_precomputed_embeddings(cache_path)
            else:
                print(f"🎯 VERIFICATION: No precomputed cache found, will compute on-the-fly")
                self.use_cache = False
        
        if not self.use_cache:
            # For trainable models or when no cache exists, do on-the-fly inference
            print(f"🎯 VERIFICATION: Using on-the-fly BERT inference")
            token_map = self.build_code_to_tokens_map()
            
            # Initialize models
            self.code_embedder = AutoModel.from_pretrained(cfg.code_embedder)
            self.linear = nn.Linear(self.code_embedder.config.hidden_size, self.cfg.token_dim)
            
            # Freeze model if requested
            if getattr(cfg, 'freeze_model', False):
                print(f"🎯 VERIFICATION: Freezing model parameters")
                for param in self.code_embedder.parameters():
                    param.requires_grad = False
            
            # Register each tensor as a buffer
            self.key_to_buffer = {}
            for key, tensor in token_map.items():
                buffer_name = f"tokens_{key}"
                self.register_buffer(buffer_name, tensor)
                self.key_to_buffer[key] = buffer_name
            
            print(f"🎯 VERIFICATION: Hidden size = {self.code_embedder.config.hidden_size}")
    
    def _get_cache_path(self):
        """Get the path to precomputed embeddings cache"""
        from pathlib import Path
        model_name = self.cfg.code_embedder.replace('/', '_')
        
        # Show the actual mapping path for clarity
        mapping_path = self.cfg.code_metadata_fp
        print(f"🎯 MAPPING: Using mapping file: {mapping_path}")
        
        # Detect which mapping is being used
        if 'codes.parquet' in mapping_path:
            # Original mapping - use original cache
            cache_dir = Path(f"MEDS_cohort/embeddings/{model_name}_original")
            print(f"🎯 CACHE: Detected ORIGINAL mapping → using cache: {cache_dir}")
        else:
            # Enhanced mapping - use enhanced cache
            cache_dir = Path(f"MEDS_cohort/embeddings/{model_name}")
            print(f"🎯 CACHE: Detected ENHANCED mapping → using cache: {cache_dir}")
        
        return cache_dir / "code_embeddings_cache.pkl"
    
    def _load_precomputed_embeddings(self, cache_path):
        """Load precomputed embeddings from cache"""
        import pickle
        import numpy as np
        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)
        
        # The embeddings are stored as a dict mapping vocab_index -> numpy array
        embeddings_dict = cache_data['embeddings']
        
        # Convert to tensor and create lookup table
        # The embeddings_dict maps vocab_index to numpy array
        # We need to create a tensor where index i corresponds to vocab_index i
        max_vocab_index = max(embeddings_dict.keys())
        embedding_dim = list(embeddings_dict.values())[0].shape[0]
        
        # Create tensor with proper indexing
        self.precomputed_embeddings = torch.zeros(max_vocab_index + 1, embedding_dim, dtype=torch.float32)
        for vocab_index, embedding in embeddings_dict.items():
            self.precomputed_embeddings[vocab_index] = torch.tensor(embedding, dtype=torch.float32)
        
        # Ensure tensor is on CPU for device compatibility
        self.precomputed_embeddings = self.precomputed_embeddings.cpu()
        
        self.linear = nn.Linear(embedding_dim, self.cfg.token_dim)
        
        print(f"🎯 CACHE: Loaded {len(embeddings_dict)} precomputed embeddings")
        print(f"🎯 CACHE: Embedding dimension = {embedding_dim}")
        print(f"🎯 CACHE: Max vocab index = {max_vocab_index}")
        print(f"🎯 CACHE: Cache tensor shape = {self.precomputed_embeddings.shape}")
        
        # Test sample embeddings to verify they're not all zeros
        # Use safe test codes that won't exceed the vocabulary size
        max_vocab = min(max_vocab_index, 20)  # Cap at 20 to be extra safe
        test_codes = [1, 2, 3, 5, 10]  # Skip code 0 (padding), use very small numbers
        print(f"🎯 CACHE: Sample embeddings for codes {test_codes} (max_vocab={max_vocab}):")
        for code in test_codes:
            if code <= max_vocab:
                embedding = self.precomputed_embeddings[code]
                is_zero = torch.allclose(embedding, torch.zeros_like(embedding))
                print(f"  Code {code}: {'ZERO' if is_zero else 'NON-ZERO'} - {embedding[:5].tolist()}")
            else:
                print(f"  Code {code}: OUT_OF_RANGE (max_vocab={max_vocab})")
        
        # Also verify that code 0 (padding) is zero as expected
        if 0 <= max_vocab:
            padding_embedding = self.precomputed_embeddings[0]
            is_padding_zero = torch.allclose(padding_embedding, torch.zeros_like(padding_embedding))
            print(f"  Code 0 (padding): {'ZERO' if is_padding_zero else 'NON-ZERO'} - {padding_embedding[:5].tolist()}")
        else:
            print(f"  Code 0 (padding): OUT_OF_RANGE (max_vocab={max_vocab})")
    
    @TimeableMixin.TimeAs
    def build_code_to_tokens_map(self):
        """Build mapping from code indices to tokenized descriptions"""
        print(f"🎯 VERIFICATION: Loading code metadata from {self.cfg.code_metadata_fp}")
        
        # Load code metadata using polars like the working encoder
        import polars as pl
        
        if self.cfg.code_metadata_fp.endswith('.csv'):
            # For CSV format, assume it has code and description columns
            code_metadata = pl.read_csv(self.cfg.code_metadata_fp)
            # Add vocab_index if not present
            if 'code/vocab_index' not in code_metadata.columns:
                code_metadata = code_metadata.with_row_index().with_columns(
                    pl.col('index').alias('code/vocab_index')
                )
        else:
            # Assume parquet format like the working encoder
            code_metadata = pl.scan_parquet(self.cfg.code_metadata_fp).select(
                ["code", "code/vocab_index", "description"]
            )
            code_metadata = code_metadata.sort("code/vocab_index").collect()
        
        # Process code names like the working encoder
        code_metadata = code_metadata.with_columns(
            pl.col("code").fill_null("").str.replace_all("//", " ").str.replace_all("_", " ")
        )
        
        # Merge code names into description when the description is missing
        code_metadata = code_metadata.with_columns(
            [
                pl.when(pl.col("description").is_null())
                .then(pl.col("code"))
                .otherwise(pl.col("description"))
                .alias("description")
            ]
        )
        
        print(f"🎯 VERIFICATION: Loaded {len(code_metadata)} code descriptions")
        
        # Tokenize all descriptions at once like the working encoder
        tokenizer = AutoTokenizer.from_pretrained(self.cfg.code_tokenizer)
        tokenized_code_metadata = tokenizer(
            ["[PAD]"] + code_metadata.select(["description"]).fill_null("").to_series().to_list(),
            **self.cfg.tokenizer_config,
        )
        
        print(f"🎯 VERIFICATION: Built tokenized mapping for {len(tokenized_code_metadata['input_ids'])} codes")
        return tokenized_code_metadata
    
    @TimeableMixin.TimeAs
    def forward(self, codes, mask):
        """
        Forward pass: compute embeddings for unique codes and map back
        """
        start_time = time.time()
        
        with torch.no_grad():
            unique_codes, inverse_indices = fast_unique_with_inverse(codes)
        
        if hasattr(self, 'use_cache') and self.use_cache:
            # Use precomputed embeddings (for frozen models)
            self.stats['cache_hits'] += 1
            print(f"🎯 CACHE: Using precomputed embeddings from cache")
            print(f"🎯 CACHE: Input codes shape: {codes.shape}, unique codes: {len(unique_codes)}")
            
            # Move precomputed embeddings to the same device as the input codes
            if self.precomputed_embeddings.device != codes.device:
                print(f"🎯 CACHE: Moving embeddings from {self.precomputed_embeddings.device} to {codes.device}")
                self.precomputed_embeddings = self.precomputed_embeddings.to(codes.device)
            
            # SAFETY CHECK: Ensure all unique codes are within bounds
            max_valid_code = self.precomputed_embeddings.shape[0] - 1
            if unique_codes.max() > max_valid_code:
                print(f"🎯 CACHE: WARNING - Found code {unique_codes.max()} but max valid is {max_valid_code}")
                print(f"🎯 CACHE: Clamping codes to valid range [0, {max_valid_code}]")
                unique_codes = torch.clamp(unique_codes, 0, max_valid_code)
            
            embeddings = self.precomputed_embeddings[unique_codes]
            print(f"🎯 CACHE: Retrieved embeddings shape: {embeddings.shape}")
            print(f"🎯 CACHE: Sample embedding (first code): {embeddings[0][:5].tolist()}...")
        else:
            # Do on-the-fly BERT inference (for trainable models)
            self.stats['bert_inferences'] += 1
            print(f"🎯 BERT: Computing embeddings on-the-fly with BERT")
            print(f"🎯 BERT: Input codes shape: {codes.shape}, unique codes: {len(unique_codes)}")
            
            # SAFETY CHECK: Ensure all unique codes are within bounds for BERT inputs
            max_valid_code = getattr(self, self.key_to_buffer['input_ids']).shape[0] - 1
            if unique_codes.max() > max_valid_code:
                print(f"🎯 BERT: WARNING - Found code {unique_codes.max()} but max valid is {max_valid_code}")
                print(f"🎯 BERT: Clamping codes to valid range [0, {max_valid_code}]")
                unique_codes = torch.clamp(unique_codes, 0, max_valid_code)
            
            # Access the tensors through their registered buffer names like the working encoder
            embedder_inputs = {
                key: getattr(self, self.key_to_buffer[key])[unique_codes] for key in self.key_to_buffer.keys()
            }
            
            print(f"🎯 BERT: Tokenized inputs shape: {embedder_inputs['input_ids'].shape}")
            print(f"🎯 BERT: Attention mask shape: {embedder_inputs['attention_mask'].shape}")
            
            # Get model embeddings
            with torch.no_grad() if getattr(self.cfg, 'freeze_model', False) else torch.enable_grad():
                outputs = self.code_embedder(**embedder_inputs)
                
                # Get embeddings (use pooler_output if available, otherwise mean pool)
                if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                    embeddings = outputs.pooler_output
                    print(f"🎯 BERT: Using pooler_output, shape: {embeddings.shape}")
                else:
                    # Mean pooling over sequence length
                    attention_mask = embedder_inputs.get('attention_mask', None)
                    if attention_mask is not None:
                        embeddings = (outputs.last_hidden_state * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
                        print(f"🎯 BERT: Using mean pooling with attention mask, shape: {embeddings.shape}")
                    else:
                        embeddings = outputs.last_hidden_state.mean(dim=1)
                        print(f"🎯 BERT: Using simple mean pooling, shape: {embeddings.shape}")
                
                print(f"🎯 BERT: Raw BERT embeddings shape: {embeddings.shape}")
                print(f"🎯 BERT: Sample raw embedding (first code): {embeddings[0][:5].tolist()}...")
        
        # Apply linear projection
        print(f"🎯 PROJECTION: Input embeddings shape: {embeddings.shape}")
        print(f"🎯 PROJECTION: Projecting from {embeddings.shape[1]} to {self.cfg.token_dim} dimensions")
        embeddings = self.linear(embeddings)
        print(f"🎯 PROJECTION: Output embeddings shape: {embeddings.shape}")
        print(f"🎯 PROJECTION: Sample projected embedding (first code): {embeddings[0][:5].tolist()}...")
        
        # Map back to original batch using inverse indices
        embeddings = embeddings[inverse_indices]
        print(f"🎯 MAPPING: Final embeddings shape: {embeddings.shape}")
        
        # 📊 Thesis Statistics
        end_time = time.time()
        duration_ms = (end_time - start_time) * 1000
        
        # Update stats
        self.stats['total_forward_passes'] += 1
        self.stats['total_duration_ms'] += duration_ms
        self.stats['total_unique_codes'] += len(unique_codes)
        
        # Print summary every 100 forward passes
        if self.stats['total_forward_passes'] % 100 == 0:
            avg_duration = self.stats['total_duration_ms'] / self.stats['total_forward_passes']
            avg_unique_codes = self.stats['total_unique_codes'] / self.stats['total_forward_passes']
            print(f"📊 SUMMARY: {self.stats['total_forward_passes']} passes, "
                  f"avg {avg_duration:.1f}ms, avg {avg_unique_codes:.1f} codes, "
                  f"cache: {self.stats['cache_hits']}, bert: {self.stats['bert_inferences']}")
        
        print(f"📊 STATS: Forward pass took {duration_ms:.1f}ms")
        print(f"📊 STATS: Unique codes processed: {len(unique_codes)}")
        
        return embeddings


class FlexibleTextCodeEncoder(nn.Module, Module, TimeableMixin):
    """
    Flexible TextCode Encoder that supports all RQ3 experiment variants
    """
    
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        
        # Configuration flags for time/value components
        self.use_time = getattr(cfg, 'use_time', True)
        self.use_value = getattr(cfg, 'use_value', True)
        
        # Always have code embedder
        self.code_embedder = FlexibleTextCodeEmbedder(cfg)
        
        # Conditional time/value embedders
        if self.use_time:
            self.date_embedder = CVE(cfg)
        if self.use_value:
            self.numeric_value_embedder = CVE(cfg)
        
        # 🎯 CONFIGURATION VERIFICATION
        print(f"🎯 CONFIG: use_time={self.use_time}, use_value={self.use_value}")
        print(f"🎯 CONFIG: mapping_file={cfg.code_metadata_fp}")
        print(f"🎯 CONFIG: frozen={getattr(cfg, 'freeze_model', False)}")
        print(f"🎯 CONFIG: model={cfg.code_embedder}")
        print(f"🎯 CONFIG: token_dim={cfg.token_dim}")
        
        # Experiment type verification
        if 'codes.parquet' in cfg.code_metadata_fp:
            print(f"🎯 EXPERIMENT: Using ORIGINAL mapping (sparse descriptions)")
            if getattr(cfg, 'freeze_model', False):
                print(f"🎯 EXPERIMENT: Cache path will include '_original' suffix")
        else:
            print(f"🎯 EXPERIMENT: Using ENHANCED mapping (rich descriptions)")
            print(f"🎯 EXPERIMENT: Cache path is default (no suffix)")
        
        if getattr(cfg, 'freeze_model', False):
            print(f"🎯 EXPERIMENT: FROZEN - using precomputed cache")
        else:
            print(f"🎯 EXPERIMENT: TRAINABLE - using on-the-fly BERT")
        
        # Component verification
        if self.use_time and self.use_value:
            print(f"🎯 COMPONENTS: Code + Time + Value = Full triplet")
        elif self.use_time:
            print(f"🎯 COMPONENTS: Code + Time only")
        elif self.use_value:
            print(f"🎯 COMPONENTS: Code + Value only")
        else:
            print(f"🎯 COMPONENTS: Code only")
    
    @TimeableMixin.TimeAs
    def get_embedding(self, batch):
        """
        Get embeddings for the input data
        """
        # Extract data from batch dict like the working encoder
        static_mask = batch["static_mask"]
        code = batch["code"]
        code_mask = batch["mask"]
        numeric_value = batch["numeric_value"]
        time_delta_days = batch["time_delta_days"]
        numeric_value_mask = batch["numeric_value_mask"]
        
        # Embed codes using our flexible embedder
        code_emb = self.code_embedder.forward(code, code_mask)
        code_emb = code_emb.permute(0, 2, 1)
        
        print(f"🎯 DATA: code_emb shape={code_emb.shape}, sample={code_emb[0,0,:5].tolist()}")
        
        # Start with code embeddings
        embedding = code_emb
        
        # Conditionally add time embeddings
        if self.use_time and hasattr(self, 'date_embedder'):
            time_emb = self.embed_func(self.date_embedder, time_delta_days) * ~static_mask.unsqueeze(dim=1)
            print(f"🎯 DATA: time_emb shape={time_emb.shape}, sample={time_emb[0,0,:5].tolist()}")
            print(f"🎯 DATA: Adding time embeddings to final embedding")
            embedding = embedding + time_emb
        elif self.use_time:
            print(f"🎯 DATA: ERROR - use_time=True but no date_embedder found")
        else:
            print(f"🎯 DATA: Skipping time embeddings (use_time=false)")
        
        # Conditionally add value embeddings
        if self.use_value and hasattr(self, 'numeric_value_embedder'):
            val_emb = self.embed_func(self.numeric_value_embedder, numeric_value) * numeric_value_mask.unsqueeze(dim=1)
            print(f"🎯 DATA: val_emb shape={val_emb.shape}, sample={val_emb[0,0,:5].tolist()}")
            print(f"🎯 DATA: Adding value embeddings to final embedding")
            embedding = embedding + val_emb
        elif self.use_value:
            print(f"🎯 DATA: ERROR - use_value=True but no numeric_value_embedder found")
        else:
            print(f"🎯 DATA: Skipping value embeddings (use_value=false)")
        
        print(f"🎯 DATA: Final embedding shape={embedding.shape}, sample={embedding[0,0,:5].tolist()}")
        
        assert embedding.isfinite().all(), "Embedding is not finite"
        if embedding.shape[-1] > self.cfg.max_seq_len:
            raise ValueError(
                f"Triplet embedding length {embedding.shape[-1]} "
                "is greater than max_seq_len {self.cfg.max_seq_len}"
            )
        return embedding.transpose(1, 2)
    
    @TimeableMixin.TimeAs
    def embed_func(self, embedder, x):
        out = embedder.forward(x[None, :].transpose(2, 0)).permute(1, 2, 0)
        return out
    
    @TimeableMixin.TimeAs
    def forward(self, batch):
        embedding = self.get_embedding(batch)
        batch[INPUT_ENCODER_MASK_KEY] = batch["mask"]
        batch[INPUT_ENCODER_TOKENS_KEY] = embedding
        return batch 