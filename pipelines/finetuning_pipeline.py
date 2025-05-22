from zenml import pipeline

from model.fine_tune_qwen2vl_step import fine_tune_qwen2vl
from model.push_to_hub import push_to_hub
from model.system_message import system_message_prompt
from steps.format_data_step import format_data_step
from steps.load_data_step import load_data_step
from steps.preprocess_data_step import preprocess_data_step
from steps.split_data_step import split_data_step


@pipeline()
def finetuning_pipeline():
    model_id = "Qwen/Qwen2-VL-2B-Instruct"
    system_message = system_message_prompt
    raw_data = load_data_step()
    preprocessed_data = preprocess_data_step(raw_data)
    formatted_data = format_data_step(preprocessed_data,system_message)
    train_set, eval_set, test_set = split_data_step(formatted_data)
    model, processor = fine_tune_qwen2vl(model_id, train_set, eval_set)
    push_to_hub(model, processor)