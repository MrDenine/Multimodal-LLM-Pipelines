from zenml import pipeline

from steps.create_dataset_visualization_step import create_dataset_visualization_step
from steps.load_data_step import load_data_step
from steps.preprocess_data_step import preprocess_data_step

@pipeline
def dataset_visualization_pipeline():
    dataset = load_data_step()
    preprocessed_data = preprocess_data_step(dataset) # Function we will see later
    create_dataset_visualization_step(preprocessed_data)