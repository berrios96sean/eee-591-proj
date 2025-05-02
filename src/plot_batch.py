import pandas as pd
import matplotlib.pyplot as plt

# Lists to store metrics
precisions = []
recalls = []
map50s = []

# Read data from the CSV
df = pd.read_csv("hyp_batch_size_yolov9.csv")

# Extract epochs from the 'epoch' column
# The 'epoch' column might be formatted as "X/Y", so we split and take the first part
if "batch_size" in df.columns:
    batch = df["batch_size"].tolist()
else:
    # Fallback: if no 'epoch' column, assume epochs are 1 to len(df)
    batch = list(range(1, len(df) + 1))

# Extract metrics for all epochs
precisions = df["precision"].tolist()
recalls = df["recall"].tolist()
map50s = df["mAP50"].tolist()

# Plot all metrics over epoch count
plt.figure(figsize=(8, 5))
plt.plot(batch, map50s, marker="o", label="mAP@50", color="green")
plt.plot(batch, precisions, marker="o", label="Precision", color="red")
plt.plot(batch, recalls, marker="o", label="Recall", color="blue")

# Customize the plot
plt.xticks([b for b in batch])
plt.xlabel("Batch Size")
plt.ylabel("Average Metric Value")
plt.title("Batch Size vs Average Metrics")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the plot
plt.savefig("bs_average_metrics_comparison_yolov9.png", dpi=300)
plt.show()
