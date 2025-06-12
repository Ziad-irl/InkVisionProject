import matplotlib.pyplot as plt

# Define tasks and their timelines (start week, duration in weeks)
tasks = [
    ("Dataset Collection & Environment Setup", 1, 2),
    ("Printed OCR Module (Tesseract/EasyOCR)", 3, 2),
    ("Handwriting OCR Module (TrOCR)", 5, 2),
    ("IAM Label Extraction & Cleaning", 6, 1),
    ("Evaluation Metrics Implementation", 7, 2),
    ("Table & Math Module Prototyping", 9, 2),
    ("Results Analysis, Paper Writing & Presentation", 11, 2)
]

# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Plot each task as a horizontal bar
for task, start, duration in tasks:
    ax.barh(task, duration, left=start, edgecolor='black')

# Customize axes
ax.set_xlabel("Week Number")
ax.set_title("InkVision Project Timeline (Weeks 1–12)")
ax.set_xlim(0, 13)
ax.set_xticks(range(1, 13))
ax.invert_yaxis()  # Highest-level task at the top

# Annotate weeks on each bar
for task, start, duration in tasks:
    ax.text(start + duration/2, task, f"Weeks {start}-{start+duration-1}", 
            va='center', ha='center', color='white', fontsize=9)

plt.tight_layout()
plt.show()
