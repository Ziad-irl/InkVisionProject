# run_full_pipeline.py

import os
import json
import pandas as pd

from src.printed_batch_runner import run_printed_batch
from src.batch_runner import run_batch as run_handwriting_batch

# ─────────────────────────────────────────────────────────────────────────────
# 1) CONFIGURATION (Change these paths if needed)
# ─────────────────────────────────────────────────────────────────────────────
PRINTED_IMAGES_DIR   = "D:/InkVisionProject/data/printed/images"
PRINTED_LABELS_FILE  = "D:/InkVisionProject/data/printed/labels.txt"

HANDWRITING_IMAGES_DIR  = "D:/InkVisionProject/data/handwriting/images"
HANDWRITING_LABELS_FILE = "D:/InkVisionProject/data/handwriting/labels.txt"

# Output files
PRINTED_CSV_OUT     = "D:/InkVisionProject/outputs/printed_results.csv"
HANDWRITING_CSV_OUT = "D:/InkVisionProject/outputs/handwriting_results.csv"
COMBINED_JSON_OUT   = "D:/InkVisionProject/outputs/full_pipeline.json"

# ─────────────────────────────────────────────────────────────────────────────
# 2) RUN PRINTED OCR + EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Running Printed OCR Batch ===")
printed_results = run_printed_batch(
    folder_path=PRINTED_IMAGES_DIR,
    label_file=PRINTED_LABELS_FILE,
    use_easyocr=True,
    use_tesseract=False  # set to True if you want Tesseract as well
)

# Convert printed_results into a DataFrame and save CSV
if printed_results:
    df_printed = pd.DataFrame(printed_results)
    df_printed.to_csv(PRINTED_CSV_OUT, index=False, encoding='utf-8')
    print(f"\n✅ Printed OCR results saved to: {PRINTED_CSV_OUT}")
else:
    print("\n⚠️ No printed results to save.")

# ─────────────────────────────────────────────────────────────────────────────
# 3) RUN HANDWRITING OCR + EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Running Handwriting OCR Batch ===")
handwriting_results = run_handwriting_batch(
    folder_path=HANDWRITING_IMAGES_DIR,
    label_file=HANDWRITING_LABELS_FILE
)

# The run_handwriting_batch() function already prints per-image metrics.
# Convert handwriting_results into a DataFrame and save CSV
if handwriting_results:
    # handwriting_results is a list of tuples: [(filename, expected, prediction, metrics),...]
    # Transform into list of dicts for DataFrame
    hw_dicts = []
    for filename, expected, prediction, metrics in handwriting_results:
        hw_dicts.append({
            "filename": filename,
            "ground_truth": expected,
            "prediction": prediction,
            "wer": metrics["WER"],
            "cer": metrics["CER"],
            "f1": metrics["F1"]
        })
    df_hand = pd.DataFrame(hw_dicts)
    df_hand.to_csv(HANDWRITING_CSV_OUT, index=False, encoding='utf-8')
    print(f"\n✅ Handwriting OCR results saved to: {HANDWRITING_CSV_OUT}")
else:
    print("\n⚠️ No handwriting results to save.")

# ─────────────────────────────────────────────────────────────────────────────
# 4) SAVE COMBINED JSON (Optional)
# ─────────────────────────────────────────────────────────────────────────────
full_output = {
    "printed": printed_results if printed_results else [],
    "handwriting": hw_dicts if handwriting_results else []
}
with open(COMBINED_JSON_OUT, "w", encoding="utf-8") as jf:
    json.dump(full_output, jf, indent=2, ensure_ascii=False)
print(f"\n✅ Full pipeline JSON saved to: {COMBINED_JSON_OUT}")

# ─────────────────────────────────────────────────────────────────────────────
# 5) SUMMARY METRICS (Optional Print‐Out)
# ─────────────────────────────────────────────────────────────────────────────
def print_summary(df, label):
    if df is None or df.empty:
        print(f"\n⚠️ No data for {label}.")
        return
    print(f"\n--- {label} Summary ---")
    # Compute mean WER/CER/F1 across all images
    avg_wer = df["wer"].mean()
    avg_cer = df["cer"].mean()
    avg_f1  = df["f1"].mean()
    print(f"Average WER: {avg_wer:.2f}%")
    print(f"Average CER: {avg_cer:.2f}%")
    print(f"Average F1 : {avg_f1:.2f}%")

try:
    df_printed = pd.read_csv(PRINTED_CSV_OUT)
    print_summary(df_printed, "Printed OCR")
except:
    pass

try:
    df_handwriting = pd.read_csv(HANDWRITING_CSV_OUT)
    print_summary(df_handwriting, "Handwriting OCR")
except:
    pass

print("\n*** Full pipeline run complete! ***\n")
