from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from zenml import pipeline
from zenml.client import Client

from steps.evaluate_step import evaluate
from steps.format_data_step import format_data_step
from steps.load_data_step import load_data_step
from steps.preprocess_data_step import preprocess_data_step
from steps.split_data_step import split_data_step
from steps.visualize_step import visualize


@pipeline()
def evaluate_pipeline():
    client = Client()
    latest_run = client.get_pipeline("finetuning_pipeline").last_run
    model_id = (latest_run.steps["push_to_hub"].outputs["output_dir"])[0].load()
    raw_data = load_data_step()
    preprocessed_data = preprocess_data_step(raw_data)
    formatted_data = format_data_step(preprocessed_data)
    train_set, eval_set, test_set = split_data_step(formatted_data)
    model = Qwen2VLForConditionalGeneration.from_pretrained(model_id)
    processor = Qwen2VLProcessor.from_pretrained(model_id)
    output = evaluate(model, processor, test_set)
    visualize(test_set, output)