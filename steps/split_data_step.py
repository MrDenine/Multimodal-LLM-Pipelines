import random
from typing import Annotated, List, Tuple
from zenml import step

@step
def split_data_step(formatted_data, train_ratio=0.70, eval_ratio=0.20, seed = 42) -> Tuple[
    Annotated[List, "train_set"],
    Annotated[List, "eval_set"],
    Annotated[List, "test_set"],
]:
    random.seed(seed)
    random.shuffle(formatted_data)  # Shuffle the dataset
    train_size = int(len(formatted_data) * train_ratio)
    eval_size = int(len(formatted_data) * eval_ratio)
    train_set = formatted_data[:train_size]
    eval_set = formatted_data[train_size:train_size + eval_size]
    test_set = formatted_data[train_size + eval_size:]
    return train_set, eval_set, test_set