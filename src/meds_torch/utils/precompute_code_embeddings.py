#!/usr/bin/env python3
"""
Precompute code-only embeddings for improved textcode encoder.

This script extracts unique codes from codes.parquet and generates embeddings 
using specified HuggingFace models. The embeddings are saved in native dimension
(not projected) so projection can be learned during training.

Usage:
    python -m meds_torch.utils.precompute_code_embeddings --model nlpie/tiny-clinicalbert
    python -m meds_torch.utils.precompute_code_embeddings --model all  # All three models
"""

import argparse
import os
import pickle
import sys
import time
import GPUtil
import psutil
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

# Add meds-torch to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def load_code_descriptions(mapping_path: str) -> Dict[int, str]:
    """Load code descriptions from mapping file."""
    code_desc = {}
    
    # Load from mapping file (authoritative descriptions)
    if os.path.exists(mapping_path):
        try:
            import csv
            with open(mapping_path, newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    try:
                        vocab_index = int(row['vocab_index'])  # Use vocab_index as key
                        desc = row['description']
                        code_desc[vocab_index] = desc
                    except Exception as e:
                        continue
            
            print(f"[Precompute] Loaded {len(code_desc)} descriptions from mapping file")
        except Exception as e:
            print(f"[Precompute] Warning: could not load mapping file ({e})")
    else:
        print(f"[Precompute] Error: Mapping file not found at {mapping_path}")
    
    return code_desc


def create_code_textualizations(code_desc: Dict[int, str], 
                               prompt_template: str = "{desc}",
                               max_desc_words: int = 1000) -> Dict[int, str]:
    """Create textualizations for codes only."""
    textualizations = {}
    
    for vocab_index, desc in code_desc.items():
        # Limit description length (0 means no limit, use full description)
        desc_words = str(desc).split()
        if max_desc_words > 0:
            desc_short = " ".join(desc_words[:max_desc_words]) if desc_words else f"Code_{vocab_index}"
        else:
            # Use full description when max_desc_words is 0
            desc_short = str(desc) if desc else f"Code_{vocab_index}"
        
        # Create textualization
        text = prompt_template.format(desc=desc_short)
        textualizations[vocab_index] = text
    
    return textualizations


def generate_code_embeddings(textualizations: Dict[int, str], 
                            model_name: str = "nlpie/tiny-clinicalbert",
                            device: str = "auto",
                            batch_size: int = 32,
                            use_all_gpus: bool = False) -> Dict[int, np.ndarray]:
    """Generate embeddings for code descriptions."""
    
    start_time = time.time()
    
    # Setup device and multi-GPU
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"[Precompute] Loading model: {model_name}")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Handle pad token for different model types
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.pad_token = tokenizer.eos_token
    
    # Determine model type and load appropriate model
    model_type = "auto"
    if any(name in model_name.lower() for name in ["bert", "roberta", "distilbert", "clinical", "bio"]):
        model_type = "encoder"
    elif any(name in model_name.lower() for name in ["gpt", "llama", "qwen", "gemma", "mistral"]):
        model_type = "decoder"
    else:
        model_type = "encoder"
    
    if model_type == "encoder":
        model = AutoModel.from_pretrained(model_name)
    elif model_type == "decoder":
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(model_name)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    # Setup multi-GPU if requested
    if use_all_gpus and torch.cuda.device_count() > 1:
        num_gpus = torch.cuda.device_count()
        print(f"[Precompute] 🚀 Using {num_gpus} GPUs for parallel processing!")
        print(f"[Precompute] Expected speedup: ~{num_gpus}x faster than single GPU")
        device = "cuda"
        # Use DataParallel for multi-GPU
        model = torch.nn.DataParallel(model)
    else:
        print(f"[Precompute] Using device: {device}")
        if torch.cuda.is_available():
            print(f"[Precompute] Available GPUs: {torch.cuda.device_count()}")
    
    model.to(device)
    model.eval()
    
    # Log system info
    print(f"[Precompute] System Info:")
    print(f"  - CPU: {psutil.cpu_count()} cores")
    print(f"  - RAM: {psutil.virtual_memory().total / 1024**3:.1f} GB")
    if torch.cuda.is_available():
        print(f"  - GPU: {torch.cuda.get_device_name(0)}")
        print(f"  - GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        try:
            gpus = GPUtil.getGPUs()
            for i, gpu in enumerate(gpus):
                print(f"  - GPU {i}: {gpu.name}, Memory: {gpu.memoryTotal}MB")
        except:
            pass
    
    model.to(device)
    model.eval()
    
    # Get hidden size
    if hasattr(model, 'module'):
        # DataParallel wraps the model
        hidden_size = model.module.config.hidden_size
    else:
        # Single GPU model
        hidden_size = model.config.hidden_size
    print(f"[Precompute] Model hidden size: {hidden_size}")
    
    # Process in batches
    embeddings = {}
    codes = list(textualizations.keys())
    texts = list(textualizations.values())
    
    print(f"[Precompute] Processing {len(texts)} codes in batches of {batch_size}")
    print(f"[Precompute] Estimated batches: {len(texts) // batch_size + 1}")
    
    batch_times = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
        batch_start = time.time()
        batch_texts = texts[i:i + batch_size]
        batch_codes = codes[i:i + batch_size]
        
        # Tokenize
        tokenized = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(device)
        
        # Generate embeddings
        with torch.no_grad():
            if model_type == "decoder":
                outputs = model(**tokenized, output_hidden_states=True)
            else:
                outputs = model(**tokenized)
            
            # Extract embeddings based on model type
            if model_type == "encoder":
                if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                    batch_embeddings = outputs.pooler_output  # (N, hidden)
                else:
                    # For encoder models without pooler, use mean pooling
                    if hasattr(outputs, 'last_hidden_state'):
                        last_hidden = outputs.last_hidden_state  # (N, seq_len, hidden)
                        if tokenized.get("attention_mask") is not None:
                            mask_expanded = tokenized["attention_mask"].unsqueeze(-1).expand(last_hidden.size())
                            masked_hidden = last_hidden * mask_expanded.float()
                            token_counts = tokenized["attention_mask"].sum(dim=1, keepdim=True).unsqueeze(-1)
                            batch_embeddings = masked_hidden.sum(dim=1) / token_counts.squeeze(-1)
                        else:
                            batch_embeddings = last_hidden.mean(dim=1)
                    else:
                        raise ValueError(f"Unknown output format for encoder model: {model_name}")
            else:
                # For decoder models, use mean pooling
                # Handle different model types
                if hasattr(outputs, 'last_hidden_state'):
                    # Encoder models (BERT, etc.)
                    last_hidden = outputs.last_hidden_state  # (N, seq_len, hidden)
                elif hasattr(outputs, 'hidden_states'):
                    # Decoder models (GPT, Qwen, etc.)
                    last_hidden = outputs.hidden_states[-1]  # (N, seq_len, hidden)
                else:
                    raise ValueError(f"Unknown output format for model: {model_name}")
                if tokenized.get("attention_mask") is not None:
                    mask_expanded = tokenized["attention_mask"].unsqueeze(-1).expand(last_hidden.size())
                    masked_hidden = last_hidden * mask_expanded.float()
                    token_counts = tokenized["attention_mask"].sum(dim=1, keepdim=True).unsqueeze(-1)
                    batch_embeddings = masked_hidden.sum(dim=1) / token_counts.squeeze(-1)
                else:
                    batch_embeddings = last_hidden.mean(dim=1)
        
        # Store embeddings
        for j, code in enumerate(batch_codes):
            embeddings[code] = batch_embeddings[j].cpu().numpy()
        
        batch_time = time.time() - batch_start
        batch_times.append(batch_time)
        
        # Log progress every 10 batches
        if len(batch_times) % 10 == 0:
            avg_batch_time = np.mean(batch_times[-10:])
            remaining_batches = (len(texts) - i - batch_size) // batch_size
            eta = remaining_batches * avg_batch_time
            print(f"[Precompute] Batch {len(batch_times)}/{len(texts) // batch_size + 1}, "
                  f"Avg batch time: {avg_batch_time:.2f}s, ETA: {eta/60:.1f}min")
    
    total_time = time.time() - start_time
    print(f"[Precompute] ✅ Embedding generation completed!")
    print(f"[Precompute] Total time: {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"[Precompute] Average batch time: {np.mean(batch_times):.2f}s")
    print(f"[Precompute] Codes processed: {len(embeddings)}")
    
    return embeddings


def main():
    overall_start_time = time.time()
    
    parser = argparse.ArgumentParser(description="Precompute code-only embeddings for improved textcode encoder")
    parser.add_argument("--log_file", type=str, default="precompute_logs.txt", 
                       help="Log file to save output")
    parser.add_argument("--model", type=str, default="nlpie/tiny-clinicalbert",
                       help="HuggingFace model name or 'all' for all three models")
    parser.add_argument("--mapping_path", type=str, default="mapping/meds_triplet_descriptions.csv", 
                       help="Path to mapping CSV file")
    parser.add_argument("--output_dir", type=str, default="../MEDS_cohort/embeddings", 
                       help="Output directory for cache files")
    parser.add_argument("--device", type=str, default="auto", 
                       help="Device to use (auto/cpu/cuda)")
    parser.add_argument("--prompt_template", type=str, default="{desc}", 
                       help="Prompt template for code textualization")
    parser.add_argument("--max_desc_words", type=int, default=1000, 
                       help="Maximum words in description (0 for no limit)")
    parser.add_argument("--batch_size", type=int, default=32, 
                       help="Batch size for GPU processing")
    parser.add_argument("--use_all_gpus", action="store_true", 
                       help="Use all available GPUs for parallel processing")
    
    args = parser.parse_args()
    
    # Setup logging to both console and file
    import sys
    from datetime import datetime
    
    # Create log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{args.log_file}_{timestamp}.txt"
    
    # Redirect stdout to both console and file
    class Logger:
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, 'w')
        
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush()
        
        def flush(self):
            self.terminal.flush()
            self.log.flush()
    
    sys.stdout = Logger(log_filename)
    
    print(f"[Precompute] 🚀 Starting code embedding precomputation...")
    print(f"[Precompute] Log file: {log_filename}")
    print(f"[Precompute] Configuration:")
    print(f"  - Model: {args.model}")
    print(f"  - Mapping path: {args.mapping_path}")
    print(f"  - Output dir: {args.output_dir}")
    print(f"  - Device: {args.device}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Max desc words: {args.max_desc_words} ({'no limit' if args.max_desc_words == 0 else f'limit to {args.max_desc_words} words'})")
    print(f"  - Use all GPUs: {args.use_all_gpus}")
    print(f"  - Prompt template: {args.prompt_template}")
    
    # Define the three models
    models = {
        "tinybert": "nlpie/tiny-clinicalbert",
        "modernbioclinicalbert": "thomas-sounack/BioClinical-ModernBERT-large",
        "qwen3": "Qwen/Qwen3-Embedding-0.6B"
    }
    
    # Determine which models to process
    if args.model == "all":
        models_to_process = models
    else:
        # Check if it's one of our predefined models
        if args.model in models:
            models_to_process = {args.model: models[args.model]}
        else:
            # Custom model
            models_to_process = {"custom": args.model}
    
    # Load code descriptions
    print(f"\n[Precompute] Loading code descriptions...")
    load_start = time.time()
    code_desc = load_code_descriptions(args.mapping_path)
    load_time = time.time() - load_start
    
    if not code_desc:
        print(f"[Precompute] No code descriptions found, exiting...")
        return
    
    print(f"[Precompute] Found {len(code_desc)} codes with descriptions")
    print(f"[Precompute] Loading time: {load_time:.2f}s")
    
    # Create textualizations
    print(f"\n[Precompute] Creating textualizations...")
    text_start = time.time()
    textualizations = create_code_textualizations(
        code_desc, args.prompt_template, args.max_desc_words
    )
    text_time = time.time() - text_start
    print(f"[Precompute] Textualization time: {text_time:.2f}s")
    
    # Log sample descriptions to verify no truncation
    print(f"\n[Precompute] 📝 Sample descriptions (verifying no truncation):")
    sample_codes = list(textualizations.items())[:10]  # First 10
    longest_desc = max(textualizations.items(), key=lambda x: len(x[1]))  # Longest description
    
    for i, (vocab_index, desc) in enumerate(sample_codes):
        word_count = len(desc.split())
        print(f"  {i+1:2d}. vocab_{vocab_index:4d}: {word_count:3d} words - {desc[:80]}{'...' if len(desc) > 80 else ''}")
    
    # Show the longest description
    longest_vocab, longest_text = longest_desc
    longest_word_count = len(longest_text.split())
    print(f"  Longest: vocab_{longest_vocab:4d}: {longest_word_count:3d} words - {longest_text[:100]}{'...' if len(longest_text) > 100 else ''}")
    
    if args.max_desc_words == 0:
        print(f"  ✅ No truncation (max_desc_words=0 means no limit)")
    else:
        print(f"  ⚠️  Limited to {args.max_desc_words} words")
    
    # Process each model
    model_times = {}
    total_cache_size = 0
    
    for model_key, model_name in models_to_process.items():
        print(f"\n[Precompute] 🚀 Processing model: {model_key} ({model_name})")
        model_start = time.time()
        
        # Setup output path
        output_dir = Path(args.output_dir) / model_name.replace("/", "_")
        cache_path = output_dir / "code_embeddings_cache.pkl"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if cache already exists
        if cache_path.exists():
            print(f"[Precompute] Cache already exists for {model_key}, skipping...")
            continue
        
        # Generate embeddings
        print(f"[Precompute] Generating embeddings for {len(textualizations)} codes...")
        embeddings = generate_code_embeddings(
            textualizations, model_name, args.device, args.batch_size, args.use_all_gpus
        )
        
        # Save cache immediately after each model
        print(f"[Precompute] 💾 Saving cache to {cache_path}...")
        save_start = time.time()
        cache_data = {
            "embeddings": embeddings,
            "textualizations": textualizations,
            "model_name": model_name,
            "model_key": model_key,
            "num_codes": len(embeddings),
            "hidden_size": list(embeddings.values())[0].shape[0] if embeddings else 0,
            "prompt_template": args.prompt_template,
            "max_desc_words": args.max_desc_words,
            "use_all_gpus": args.use_all_gpus,
            "batch_size": args.batch_size,
            "timestamp": time.time()
        }
        
        with open(cache_path, "wb") as f:
            pickle.dump(cache_data, f)
        
        save_time = time.time() - save_start
        model_time = time.time() - model_start
        model_times[model_key] = model_time
        cache_size = cache_path.stat().st_size / 1024 / 1024  # MB
        total_cache_size += cache_size
        
        print(f"[Precompute] ✅ Cache saved for {model_key}!")
        print(f"[Precompute] Cache contains {len(embeddings)} code embeddings")
        print(f"[Precompute] Cache size: {cache_size:.1f} MB")
        print(f"[Precompute] Model processing time: {model_time:.1f}s ({model_time/60:.1f}min)")
        print(f"[Precompute] Save time: {save_time:.2f}s")
        print(f"[Precompute] 🎯 Model {model_key} completed successfully!")
    
    # Final summary
    overall_time = time.time() - overall_start_time
    print(f"\n[Precompute] 🎉 All code embedding caches completed!")
    print(f"\n[Precompute] 📊 Final Summary:")
    print(f"  - Total time: {overall_time:.1f}s ({overall_time/60:.1f}min)")
    print(f"  - Models processed: {len(model_times)}")
    print(f"  - Total cache size: {total_cache_size:.1f} MB")
    if model_times:
        print(f"  - Average time per model: {np.mean(list(model_times.values())):.1f}s")
    else:
        print(f"  - Average time per model: N/A (no models processed)")
    print(f"  - Codes processed: {len(textualizations)}")
    
    if model_times:
        print(f"\n[Precompute] 📈 Per-model breakdown:")
        for model_key, model_time in model_times.items():
            print(f"  - {model_key}: {model_time:.1f}s ({model_time/60:.1f}min)")


if __name__ == "__main__":
    main() 