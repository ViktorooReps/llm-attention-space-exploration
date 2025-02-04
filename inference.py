import argparse
import torch

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
# For meta-llama/Llama-3.1-8B we load the model in bfloat16 precision.
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
# These are passed when tokenizing text to force truncation at 128k tokens for meta-llama.
TOKENIZER_CALL_KWARGS = {
    "meta-llama/Llama-3.1-8B": {
         "truncation": True,
         "max_length": 64000
    }
}

@torch.no_grad()
def create_representations(dataset_name: str, model_name: str, *, n_tokens: int):
    # COLLECT TEXT DATA TO PROCESS
    dataset = get_dataset(dataset_name)
    tokenizer = get_tokenizer(model_name)
    # Retrieve any extra kwargs to pass to the tokenizer when encoding text.
    tokenizer_call_kwargs = TOKENIZER_CALL_KWARGS.get(model_name, {})

    collected_encodings = []
    total_len = 0
    min_length = 99999999
    max_length = 0
    for example in dataset:
        # Pass model-specific tokenization kwargs (e.g., truncation for meta-llama).
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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    print(f'Model device: {device}')
    model.eval()

    # PROCESS THE DATA
    for encoding_id, encoding in tqdm(enumerate(collected_encodings), total=len(collected_encodings)):
        example_n_tokens = len(encoding['input_ids'])
        # Convert lists to torch tensors, add a batch dimension, and move to device.
        encoding = {key: torch.tensor(val).to(device).unsqueeze(0) for key, val in encoding.items()}

        out: ModelOutput = model(**encoding, use_cache=True)
        if 'past_key_values' not in out:
            raise RuntimeError("The model did not return past_key_values.")

        kv_cache = out['past_key_values']
        num_layers = len(kv_cache)
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
        description="Generate representations from a model based on a dataset."
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

    create_representations(dataset_name=args.dataset, model_name=args.model, n_tokens=args.n_tokens)

if __name__ == '__main__':
    main()
