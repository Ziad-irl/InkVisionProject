import os

def extract_line_labels(lines_txt_path, output_file, available_images):
    with open(lines_txt_path, "r", encoding="utf-8") as f, open(output_file, "w", encoding="utf-8") as out:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split(" ", 1)
            if len(parts) != 2:
                continue
            img_name, text = parts
            if img_name + ".png" in available_images:
                out.write(f"{img_name}.png\t{text.strip()}\n")

# Paths
lines_txt = "D:/InkVisionProject/data/handwriting/ascii/lines.txt"
image_folder = "D:/InkVisionProject/data/handwriting/images"
output_labels = "D:/InkVisionProject/data/handwriting/labels.txt"

available = set(os.listdir(image_folder))
extract_line_labels(lines_txt, output_labels, available)

print("✅ Created labels.txt with matching line images.")
