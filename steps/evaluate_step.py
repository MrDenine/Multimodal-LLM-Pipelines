from typing import Annotated, List

from qwen_vl_utils import process_vision_info
from zenml import step


@step
def evaluate(model, processor, sample, max_new_tokens=1024) -> Annotated[List, "response"]:
    text_input = processor.apply_chat_template(
        sample, tokenize=False, add_generation_prompt=True
    )
    sample = sample[0][0:2]
    image_inputs, _ = process_vision_info(sample)

    model_inputs = processor(
        text=[text_input],
        images=image_inputs,
        return_tensors="pt",
    )

    generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens)

    trimmed_generated_ids = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)]

    output_text = processor.batch_decode(
        trimmed_generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text