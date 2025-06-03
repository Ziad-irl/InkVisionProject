
# 🧠 InkVision: A Unified OCR Pipeline for Printed, Handwritten, Table, and Math Text Recognition

![InkVision Banner](https://raw.githubusercontent.com/Ziad-irl/inkvisionproject/main/assets/banner.png)

> 📌 InkVision is a modular OCR pipeline that integrates multiple state-of-the-art OCR models to extract printed, handwritten, tabular, and mathematical expressions from images with high accuracy.

---

## 📦 Features

- ✅ **Printed Text Recognition** using Tesseract OCR and EasyOCR
- ✍️ **Handwriting Recognition** using TrOCR (Transformer-based)
- 📐 **Table Extraction** using CascadeTabNet or YOLO-based approaches
- 🧮 **Math Expression Parsing** via Im2LaTeX or Donut
- 📊 **Evaluation Dashboard** with WER, CER, and F1-Score comparison
- 🔁 Modular pipeline that allows easy benchmarking between tools
- 🧪 Dataset evaluation with IAM, IIIT5K, and PubLayNet

---

## 🚀 Project Structure

```
InkVision/
│
├── data/                      # Contains datasets (printed, handwritten, tables)
├── src/                       # OCR scripts for each module
│   ├── ocr_print.py
│   ├── ocr_hand.py
│   ├── ocr_table.py
│   ├── ocr_math.py
│   ├── evaluate.py            # Computes WER/CER/F1
│   └── batch_runner.py
│
├── output/                    # Predicted outputs
├── results/                   # Evaluation metrics and visualizations
├── assets/                    # Sample results and README images
├── requirements.txt           # Required packages
├── main.py                    # Main execution script
└── README.md
```

---

## 🖥️ Models Used

| Model         | Purpose                  | Architecture      |
|---------------|---------------------------|--------------------|
| Tesseract OCR | Printed OCR               | LSTM-based         |
| EasyOCR       | Fast printed & light handwriting OCR | CNN+CTC |
| TrOCR         | Handwritten OCR           | Vision Transformer |
| CascadeTabNet | Table Detection           | Cascade RCNN       |
| Im2LaTeX      | Math OCR                  | CNN + Attention    |

---

## 📚 Datasets

| Dataset     | Domain         | Description                          |
|-------------|----------------|--------------------------------------|
| IIIT5K      | Printed words  | Word-level OCR (cropped images)      |
| IAM         | Handwriting    | Line-level handwritten text          |
| CVL         | Handwriting    | Writer-based text recognition        |
| PubLayNet   | Table layout   | Structured documents + tables        |
| CROHME      | Math           | Online handwritten math expressions  |

---

## 🧰 Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/InkVision.git
cd InkVision
```

### Step 2: Create Environment

```bash
conda create -n inkvision python=3.11
conda activate inkvision
pip install -r requirements.txt
```

### Step 3: Install Tesseract

Make sure Tesseract is installed and added to your path:

- **Windows**: [Download from here](https://github.com/tesseract-ocr/tesseract)
- **Linux**: `sudo apt install tesseract-ocr`

Set the path in `ocr_print.py` if needed:

```python
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

---

## 🏃 Running the Pipeline

### Run Main Script:

```bash
python main.py
```

This will run printed + handwritten OCR and output predictions to `/output`.

---

## 🧪 Evaluation

Metrics used:
- **WER** (Word Error Rate)
- **CER** (Character Error Rate)
- **F1 Score**
- Precision & Recall

To evaluate:

```bash
python src/evaluate.py
```

Visualized metrics and prediction vs. ground truth samples will be saved to `/results`.

---

## 📜 Citation & Credits

```bibtex
@misc{inkvision2025,
  author       = {Ziad M. Hanafi et al.},
  title        = {InkVision: Modular OCR Pipeline for Printed, Handwritten, Table, and Math Recognition},
  year         = {2025},
  note         = {https://github.com/Ziad-irl/InkVision}
}
```

---

## 🔗 References & Links

- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
- [EasyOCR](https://github.com/JaidedAI/EasyOCR)
- [TrOCR](https://huggingface.co/microsoft/trocr-base-handwritten)
- [CascadeTabNet](https://github.com/DevashishPrasad/CascadeTabNet)
- [Im2LaTeX](https://github.com/harvardnlp/im2markup)
- [IAM Dataset](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database)

---

## 👥 Contributors

- **Ziad Mahmoud Hanafi** — Printed + Pipeline + Main Logic  
- **Aliaa Maamoun** — Dataset + Result Analysis  
- **Fatma Mahmoud** — TrOCR Setup + Evaluation  
- **Abdelrhman Karam** — Metrics + Batch Testing  
- **Kamal Elsawah** — Math/Tabular OCR

---

> 🧠 “InkVision: Because no text—printed, scribbled, or structured—should go unread.”
