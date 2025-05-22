from zenml import step

@step
def preprocess_data_step(raw_data):
    # Simplified preprocessing function
    return [
        {
            'image_url': entry['image_url'],
            'question': entry['question'],
            'answer': entry[entry['answer']]
        }
        for entry in raw_data
    ]