from datasets import load_dataset
from zenml import step

@step
def load_data_step():
    return load_dataset("ccvl/3DSRBench")["test"] # The dataset has only test set.