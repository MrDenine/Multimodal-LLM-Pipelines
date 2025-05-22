from zenml import step, log_metadata
from steps.utils import format_data

@step
def format_data_step(raw_data, system_message:str):
    log_metadata(metadata={"system_message": system_message})
    return [format_data(sample,system_message) for sample in raw_data]