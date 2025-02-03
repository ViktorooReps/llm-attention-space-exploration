from pathlib import Path

import numpy as np


def format_number(value: int, max_value: int) -> str:
    num_digits = len(str(max_value))
    formatted_value = f"{value:0{num_digits}d}"
    return formatted_value


def save_representations(
        representations,
        *,
        dataset_name: str,
        model_name: str,
        layer_idx: int,
        example_idx: int,
        suffix: str = '',
        total_layers: int = 1000,
        total_examples: int = 1000000
) -> Path:
    dataset_name_normalized = dataset_name.replace('/', '__')
    model_name_normalized = model_name.replace('/', '__')

    save_path = Path('data').joinpath(
        dataset_name_normalized,
        model_name_normalized,
        format_number(layer_idx, total_layers - 1),
        f'{format_number(example_idx, total_examples - 1)}{suffix}.dat'
    )
    save_path.mkdir(parents=True, exist_ok=True)
    representations_np = np.memmap(save_path, mode='w+', shape=representations.shape, dtype=np.float16)
    representations_np[:] = representations.cpu().detach().numpy().astype(np.float16)

    return save_path


def load_representations():
    raise NotImplementedError
