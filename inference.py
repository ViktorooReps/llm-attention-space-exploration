import argparse
import os
import torch
import torch.distributed as dist

from datasets import load_dataset, Dataset, DownloadMode
from transformers import PreTrainedModel, PreTrainedTokenizer, AutoModel, AutoTokenizer, Cache
from transformers.utils import ModelOutput
from tqdm import tqdm

from datamodel import save_representations


# Dataset-specific kwargs
DATASET_KWARGS = {
    'HuggingFaceFW/fineweb-edu': {
        'split': 'train',
        'name': 'sample-10BT',
        'download_mode': DownloadMode.FORCE_REDOWNLOAD
    },
    'deepmind/pg19': {
        'split': 'train',
        'download_mode': DownloadMode.FORCE_REDOWNLOAD
    }
}


def get_dataset(dataset_name: str) -> Dataset:
    return load_dataset(dataset_name, **DATASET_KWARGS.get(dataset_name, {}), streaming=True)


# Model-specific keyword arguments.
# For meta-llama/Llama-3.1-8B we load the model in float16 precision.
MODEL_KWARGS = {
    "meta-llama/Llama-3.1-8B": {
        "torch_dtype": torch.float16,
    }
}


def get_model(model_name: str) -> PreTrainedModel:
    return AutoModel.from_pretrained(model_name, **MODEL_KWARGS.get(model_name, {}))


# Tokenizer initialization kwargs.
# For meta-llama/Llama-3.1-8B we set the tokenizer’s maximum length attribute.
TOKENIZER_INIT_KWARGS = {
    "meta-llama/Llama-3.1-8B": {
         "model_max_length": 128000
    }
}


def get_tokenizer(model_name: str) -> PreTrainedTokenizer:
    return AutoTokenizer.from_pretrained(model_name, **TOKENIZER_INIT_KWARGS.get(model_name, {}))


# Tokenizer call kwargs.
# These are passed when tokenizing text.
TOKENIZER_CALL_KWARGS = {
    "meta-llama/Llama-3.1-8B": {
         "truncation": True,
         "max_length": 128000
    }
}


def setup_distributed():
    """
    Initialize the default process group for distributed inference using NCCL.
    Sets the CUDA device based on LOCAL_RANK.
    """
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank


@torch.no_grad()
def create_representations(dataset_name: str, model_name: str, *, n_tokens: int):
    # COLLECT TEXT DATA TO PROCESS
    dataset = get_dataset(dataset_name)
    tokenizer = get_tokenizer(model_name)
    tokenizer_call_kwargs = TOKENIZER_CALL_KWARGS.get(model_name, {})

    collected_encodings = []
    total_len = 0
    min_length = 99999999
    max_length = 0
    for example in dataset:
        encoding = tokenizer(example['text'], **tokenizer_call_kwargs)
        length = len(encoding['input_ids'])
        total_len += length
        min_length = min(min_length, length)
        max_length = max(max_length, length)
        collected_encodings.append(encoding)
        if total_len >= n_tokens:
            break

    print(f'Loaded {len(collected_encodings)} examples')
    print(f'Average length: {total_len / len(collected_encodings):.2f}')
    print(f'Maximum length: {max_length}')
    print(f'Minimum length: {min_length}')

    # PREPARE THE MODEL
    model = get_model(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # If running in a distributed setting, wrap the model with FSDP.
    if dist.is_initialized():
        from torch.distributed.fsdp.wrap import wrap
        model = wrap(model)

    # Move wrapped model to device (this call is safe even if not distributed)
    model = model.to(device)
    model.eval()

    # Determine current rank (default to 0 in single-process mode)
    current_rank = dist.get_rank() if dist.is_initialized() else 0

    # PROCESS THE DATA
    for encoding_id, encoding in tqdm(enumerate(collected_encodings), total=len(collected_encodings)):
        example_n_tokens = len(encoding['input_ids'])
        # Convert token lists to tensors, add a batch dimension, and move to device.
        encoding = {key: torch.tensor(val).to(device).unsqueeze(0) for key, val in encoding.items()}

        out: ModelOutput = model(**encoding, use_cache=True)
        if 'past_key_values' not in out:
            raise RuntimeError("The model did not return past_key_values.")

        kv_cache = out['past_key_values']
        num_layers = len(kv_cache)
        # Only rank 0 writes output to avoid duplicates.
        if current_rank == 0:
            for layer_idx, (keys, values) in enumerate(kv_cache):
                batch_size, num_heads, seq_length, head_dim = keys.shape
                # Ensure batch size is 1 and the sequence length matches.
                assert batch_size == 1 and seq_length == example_n_tokens

                save_representations(
                    keys,
                    dataset_name=dataset_name,
                    model_name=model_name,
                    layer_idx=layer_idx,
                    example_idx=encoding_id,
                    suffix='_keys',
                    total_layers=num_layers,
                    total_examples=len(collected_encodings),
                )

                save_representations(
                    values,
                    dataset_name=dataset_name,
                    model_name=model_name,
                    layer_idx=layer_idx,
                    example_idx=encoding_id,
                    suffix='_values',
                    total_layers=num_layers,
                    total_examples=len(collected_encodings),
                )


def main():
    parser = argparse.ArgumentParser(
        description="Generate representations from a model based on a dataset using FSDP for multi-GPU inference."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="The name of the dataset to use (e.g., 'HuggingFaceFW/fineweb-edu')."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="The name or path of the model to use (e.g., 'meta-llama/Llama-3.1-8B')."
    )
    parser.add_argument(
        "--n_tokens",
        type=int,
        required=True,
        help="Total number of tokens to process."
    )
    args = parser.parse_args()

    # Initialize distributed processing if environment variables indicate multi-GPU mode.
    if "LOCAL_RANK" in os.environ or "RANK" in os.environ:
        local_rank = setup_distributed()
        print(f"Distributed processing initialized on local rank {local_rank}.")
    else:
        print("Running in single-process mode.")

    create_representations(dataset_name=args.dataset, model_name=args.model, n_tokens=args.n_tokens)


if __name__ == '__main__':
    main()
