# visualize_binary_confusion.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_binary_confusion(df, model_name, output_path=None):
    """
    Given a DataFrame with columns [model, TP, FP, FN, TN], plot a 2x2 confusion matrix for that model.
    """
    row = df[df["model"] == model_name].iloc[0]
    # Build 2x2 array:
    #     [[TP, FP],
    #      [FN, TN]]
    cm = np.array([[row["TP"], row["FP"]],
                   [row["FN"], row["TN"]]])
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Pred Correct", "Pred Incorrect"],
                yticklabels=["Actual Correct", "Actual Incorrect"])
    plt.title(f"Binary Confusion Matrix for {model_name}")
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
    plt.show()

if __name__ == "__main__":
    # 1) Load printed confusion CSV
    df_printed = pd.read_csv("D:/InkVisionProject/outputs/Confusions/printed_confusion.csv")
    # 2) Plot for each model
    for model in df_printed["model"].unique():
        out_png = f"D:/InkVisionProject/outputs/{model}_printed_confusion.png"
        plot_binary_confusion(df_printed, model, output_path=out_png)

    # 3) Load handwriting confusion CSV
    df_hand = pd.read_csv("D:/InkVisionProject/outputs/Confusions/handwriting_confusion.csv")
    for model in df_hand["model"].unique():
        out_png = f"D:/InkVisionProject/outputs/{model}_handwriting_confusion.png"
        plot_binary_confusion(df_hand, model, output_path=out_png)

    print("✅ Binary confusion heatmaps saved in outputs/")
