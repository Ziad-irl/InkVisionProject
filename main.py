import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from ocr_print import ocr_with_tesseract, ocr_with_easyocr
from ocr_hand import read_handwritten
import ocr_hand
from batch_runner import run_batch

run_batch(
    folder_path="D:/InkVisionProject/data/handwriting/images",
    label_file="D:/InkVisionProject/data/handwriting/labels.txt"
)

print(dir(ocr_hand))
# filepath: d:\InkVisionProject\src\ocr_print.py
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'D:\Program Files\tesseract\tesseract.exe'
# ...existing code...

# Change this to a real image path later
image_path = "D:/InkVisionProject/data/printed/sample1.PNG"
# Test the OCR functions

print("========== Tesseract OCR ==========")
print(ocr_with_tesseract(image_path))

print("\n========== EasyOCR ==========")
print(ocr_with_easyocr(image_path))
print("\n========== Handwritten OCR ==========")
image_path_hand = "D:/InkVisionProject/data/handwriting/a01-011u.PNG"
print("========== TrOCR (Handwriting) ==========")
print(read_handwritten(image_path_hand))