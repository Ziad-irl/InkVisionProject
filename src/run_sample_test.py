# run_sample_test.py

import cv2
import torch
import pytesseract
import matplotlib.pyplot as plt

from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from evaluate import evaluate_prediction

# ===== CONFIG =====
IMAGE_PATH = "data/test_samples/sample_handwritten.jpg"
GROUND_TRUTH_TEXT = "Dr. Ahmed is the Best he"
OCR_TYPE = "handwritten"  # or "printed"

# ===== OCR FUNCTIONS =====
def run_tesseract(image_path):
    image = cv2.imread(image_path)
    return pytesseract.image_to_string(image)

def run_trocr(image_path):
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten").to("cuda" if torch.cuda.is_available() else "cpu")
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixel_values = processor(images=image_rgb, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(model.device)

    generated_ids = model.generate(pixel_values, max_length=64)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

# ===== MAIN PIPELINE =====
if OCR_TYPE == "printed":
    prediction = run_tesseract(IMAGE_PATH)
else:
    prediction = run_trocr(IMAGE_PATH)

# Clean prediction
prediction = prediction.strip().replace("\n", " ")
expected = GROUND_TRUTH_TEXT.strip()

# Print Results
print("🔍 Ground Truth :", expected)
print("🧠 Prediction   :", prediction)

# Evaluate
metrics = evaluate_prediction(prediction, expected)
print(f"\n📊 Metrics — WER: {metrics['WER']:.2f}% | CER: {metrics['CER']:.2f}% | F1: {metrics['F1']:.2f}%")

# Visualize
image = cv2.imread(IMAGE_PATH)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(10, 6))
plt.imshow(image)
plt.axis("off")
plt.title(f"Prediction: {prediction}\nWER: {metrics['WER']:.1f}% | CER: {metrics['CER']:.1f}% | F1: {metrics['F1']:.1f}%", fontsize=10)
plt.tight_layout()
plt.show()
