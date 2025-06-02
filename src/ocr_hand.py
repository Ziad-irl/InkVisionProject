# D:\InkVisionProject\src\ocr_hand.py

from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
import os

# Load processor and model once
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def read_handwritten(image_path):
    """Recognize handwritten text using TrOCR."""
    if not os.path.exists(image_path):
        return "[Error] File not found:", image_path

    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

    with torch.no_grad():
        generated_ids = model.generate(pixel_values)
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return text.strip()
