import base64
from typing import Dict
from zenml import step
from zenml.types import HTMLString
from steps.utils import fetch_image

@step
def create_dataset_visualization_step(sample_data: list[Dict], sample_size=10) -> HTMLString:
    """Creates a visualization of dataset samples and returns embedded HTML."""
    html_content = """
    <div style="text-align: center; font-family: Arial, sans-serif;">
    <h2>Dataset Visualization</h2>
    """
    for entry in sample_data[1:sample_size:2]:
        image_data = fetch_image(entry['image_url'])
        image_base64 = base64.b64encode(image_data).decode('utf-8') if image_data else None
        image_tag = f'<img src="data:image/png;base64,{image_base64}" style="max-width: 300px; height: auto;">' if image_base64 else '<p>[Image not available]</p>'

        html_content += f"""
        <div style="border: 1px solid #ddd; padding: 10px; margin: 10px; display: inline-block; text-align: left;">
            {image_tag}
            <p><strong>Question:</strong> {entry['question']}</p>
            <p><strong>Answer:</strong> {entry['answer']}</p>
        </div>
        """

    html_content += "</div>"
    return HTMLString(html_content)