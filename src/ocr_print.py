import cv2
import pytesseract
import easyocr

def ocr_with_tesseract(image_path):
    """Recognizes printed text using Tesseract OCR."""
    img = cv2.imread(image_path)
    if img is None:
        return "[Error] Could not load image:", image_path
    text = pytesseract.image_to_string(img)
    return text.strip()

def ocr_with_easyocr(image_path):
    """Recognizes printed text using EasyOCR (GPU recommended)."""
    reader = easyocr.Reader(['en'], gpu=True)
    result = reader.readtext(image_path)
    return ' '.join([line[1] for line in result])
