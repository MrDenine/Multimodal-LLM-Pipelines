import requests
from zenml import log_metadata
from pynvml import *

def fetch_image(image_url):
    """Fetch image from URL and return it as a PIL image object."""
    try:
        response = requests.get(image_url, stream=True, timeout=5)
        response.raise_for_status()
        return response.content
    except requests.exceptions.RequestException:
        return None

def format_data(sample, system_message:str):
    log_metadata(metadata={"system_message": system_message})
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_message}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": sample["image_url"],
                },
                {
                    "type": "text",
                    "text": sample["question"],
                },
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": sample["answer"]}],
        },
    ]

def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")

def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()