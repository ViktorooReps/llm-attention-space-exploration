from datasets import load_dataset, Dataset
from safetensors import torch
from transformers import PreTrainedModel, PreTrainedTokenizer, AutoModel, AutoTokenizer, Cache
from transformers.utils import ModelOutput

from datamodel import save_representations

DATASET_KWARGS = {
    'HuggingFaceFW/fineweb-edu': {
        'split': 'train'
    }
}


def get_dataset(dataset_name: str) -> Dataset:
    return load_dataset(dataset_name, **DATASET_KWARGS.get(dataset_name, {}), streaming=True)


MODEL_KWARGS = {

}


def get_model(model_name: str) -> PreTrainedModel:
    return AutoModel.from_pretrained(model_name, **MODEL_KWARGS.get(model_name, {}))


TOKENIZER_KWARGS = {

}


def get_tokenizer(model_name: str) -> PreTrainedTokenizer:
    return AutoTokenizer.from_pretrained(model_name, **TOKENIZER_KWARGS.get(model_name, {}))


def create_representations(dataset_name: str, model_name: str, *, n_tokens: int):
    # COLLECT TEXT DATA TO PROCESS

    dataset = get_dataset(dataset_name)
    tokenizer = get_tokenizer(model_name)

    collected_encodings = []
    total_len = 0
    for text in dataset:
        encoding = tokenizer(text)
        total_len += len(encoding['input_ids'])
        collected_encodings.append(encoding)

        if total_len >= n_tokens:
            break

    # PREPARE THE MODEL

    model = get_model(model_name)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    model.eval()

    # PROCESS THE DATA

    for encoding_id, encoding in enumerate(collected_encodings):
        n_tokens = len(encoding['input_ids'])

        out: ModelOutput = model(**encoding.to(device), use_cache=True)
        if 'past_key_values' not in out:
            raise RuntimeError()

        kv_cache: Cache = out['past_key_values']
        assert hasattr(kv_cache, 'value_cache') and hasattr(kv_cache, 'key_cache')

        num_layers = len(kv_cache.key_cache)
        for layer_idx, (keys, values) in enumerate(zip(kv_cache.key_cache, kv_cache.value_cache)):
            batch_size, num_heads, seq_length, head_dim = keys.shape
            assert batch_size == 1 and seq_length == n_tokens

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
