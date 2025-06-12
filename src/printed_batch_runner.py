# src/printed_batch_runner.py

import os
from src.ocr_print import ocr_with_tesseract, ocr_with_easyocr
from src.evaluate import evaluate_prediction

def run_printed_batch(folder_path, label_file, use_easyocr=True, use_tesseract=False):
    """
    Run OCR on printed images (e.g. IIIT5K) using EasyOCR and/or Tesseract,
    then evaluate against ground truth labels. Returns a list of result dicts.
    """

    if not os.path.exists(label_file):
        print("❌ Printed label file not found:", label_file)
        return []

    # Read label file: each line = "filename\ttranscription"
    with open(label_file, "r", encoding="utf-8") as f:
        gt_lines = [line.strip().split("\t", 1) for line in f if "\t" in line]

    results = []
    for filename, expected in gt_lines:
        image_path = os.path.join(folder_path, filename)

        if not os.path.exists(image_path):
            print(f"[Skip] Printed image not found: {filename}")
            continue

        # Decide which OCR to run
        predictions = {}
        if use_easyocr:
            try:
                pred_easy = ocr_with_easyocr(image_path)
            except Exception as e:
                pred_easy = f"[Error EasyOCR] {e}"
            predictions["EasyOCR"] = pred_easy

        if use_tesseract:
            try:
                pred_tess = ocr_with_tesseract(image_path)
            except Exception as e:
                pred_tess = f"[Error Tesseract] {e}"
            predictions["Tesseract"] = pred_tess

        # Evaluate each prediction
        for model_name, pred_text in predictions.items():
            metrics = evaluate_prediction(pred_text, expected)
            results.append({
                "model": model_name,
                "filename": filename,
                "ground_truth": expected,
                "prediction": pred_text,
                "wer": metrics["WER"],
                "cer": metrics["CER"],
                "f1": metrics["F1"]
            })

            # Print summary per image per model
            print(f"\n🖼️ Printed {filename} → {model_name}")
            print(f"  GT : {expected}")
            print(f"  Pred: {pred_text}")
            print(f"  WER: {metrics['WER']}% | CER: {metrics['CER']}% | F1: {metrics['F1']}%")

    return results
