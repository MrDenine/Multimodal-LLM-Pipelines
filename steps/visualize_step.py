import base64

from zenml import step
from zenml.types import HTMLString

from steps.utils import fetch_image


@step
def visualize(sample, output_text) -> HTMLString:
    sample = sample[0][0:2]
    image_url = sample[1]["content"][0]["image"]
    image_data = fetch_image(image_url)
    image_base64 = base64.b64encode(image_data).decode('utf-8') if image_data else None
    image_tag = f'<img src="data:image/png;base64,{image_base64}" style="max-width: 300px; height: auto;">' if image_base64 else '<p>[Image not available]</p>'

    html_content = f"""
    <div style="text-align: center; font-family: Arial, sans-serif;">
        <h2>Evaluation Result</h2>
        <div style="border: 1px solid #ddd; padding: 10px; margin: 10px; display: inline-block; text-align: left;">
            {image_tag}
            <p><strong>Question:</strong> {sample[1]["content"][1]["text"]}</p>
            <p><strong>Answer:</strong> {output_text[0]}</p>
        </div>
    </div>
    """

    return HTMLString(html_content)