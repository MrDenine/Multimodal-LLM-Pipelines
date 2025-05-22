from zenml import step


@step
def push_to_hub(model, processor, output_dir: str = "qwen2vl-model-2b-instruct-spatial-information"):
  model.push_to_hub(output_dir)
  processor.push_to_hub(output_dir)