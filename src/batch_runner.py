# src/batch_runner.py

import os
from src.ocr_hand import read_handwritten
from src.evaluate import evaluate_prediction

def run_batch(folder_path, label_file):
    """
    Run OCR on a batch of handwriting images and evaluate performance using WER, CER, and F1.
    """
    if not os.path.exists(label_file):
        print("❌ Label file not found:", label_file)
        return

    with open(label_file, "r", encoding="utf-8") as f:
        gt_lines = [line.strip().split("\t", 1) for line in f if "\t" in line]

    results = []

    for filename, expected in gt_lines:
        image_path = os.path.join(folder_path, filename)

        if not os.path.exists(image_path):
            print(f"[Skip] {filename} not found.")
            continue

        try:
            prediction = read_handwritten(image_path)
        except Exception as e:
            print(f"[Error] Failed to process {filename}: {e}")
            continue

        metrics = evaluate_prediction(prediction, expected)

        results.append((filename, expected, prediction, metrics))

        print(f"\n🖼️ {filename}")
        print(f"📌 Ground Truth: {expected}")
        print(f"🧠 Prediction  : {prediction}")
        print(f"📊 WER: {metrics['WER']}% | CER: {metrics['CER']}% | F1: {metrics['F1']}%")

    return results
