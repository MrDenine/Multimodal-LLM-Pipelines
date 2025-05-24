from zenml import pipeline
from zenml.config import DockerSettings
from zenml.integrations.constants import HUGGINGFACE
from zenml.integrations.huggingface.services import HuggingFaceServiceConfig
from zenml.integrations.huggingface.steps import (
    huggingface_model_deployer_step,
)

docker_settings = DockerSettings(
    required_integrations=[HUGGINGFACE],
)

@pipeline(enable_cache=True, settings={"docker": docker_settings})
def huggingface_deployment_pipeline(
    repository: str,
    model_name: str,
    namespace: str,
    timeout: int = 1200,
    accelerator="gpu",
    instance_size="x1",
    instance_type="nvidia-t4",
    region="us-east-1",
    framework="pytorch",
    vendor="aws",
    task="image-text-to-text",
):
    service_config = HuggingFaceServiceConfig(repository=repository, accelerator=accelerator, instance_size=instance_size, instance_type=instance_type, region=region, vendor=vendor, framework=framework, task=task, namespace=namespace)

    huggingface_model_deployer_step(
        service_config=service_config,
        timeout=timeout,
    )