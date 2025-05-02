import pandas as pd
import matplotlib.pyplot as plt

# Define your CSV paths and their corresponding learning rate labels
csv_files = {
    "0.001": "runs/detect/train15/results.csv",
    "0.005": "runs/detect/train153/results.csv",
    "0.01":  "runs/detect/train1533/results.csv",
    "0.05":  "runs/detect/train15333/results.csv"
}

# Store results
learning_rates = [0.001,0.005,0.01,0.05]
precisions = []
recalls = []
map50s = []

# Read data from each CSV
for lr, path in csv_files.items():
    df = pd.read_csv(path)
    last_row = df.iloc[-1]  # Or df[df["metrics/mAP_0.5"] == df["metrics/mAP_0.5"].max()].iloc[0] for best epoch
    precisions.append(last_row["metrics/precision(B)"])
    recalls.append(last_row["metrics/recall(B)"])
    map50s.append(last_row["metrics/mAP50(B)"])

# Sort by learning rate
sorted_indices = sorted(range(len(learning_rates)), key=lambda i: learning_rates[i])
learning_rates = [learning_rates[i] for i in sorted_indices]
precisions = [precisions[i] for i in sorted_indices]
recalls = [recalls[i] for i in sorted_indices]
map50s = [map50s[i] for i in sorted_indices]

# Plot all metrics over learning rate
plt.figure(figsize=(8, 5))
plt.plot(learning_rates, map50s, marker='o', label='mAP@50')
plt.plot(learning_rates, precisions, marker='s', label='Precision', color='green')
plt.plot(learning_rates, recalls, marker='^', label='Recall', color='red')
plt.xticks(ticks=learning_rates, labels=[str(lr) for lr in learning_rates])
plt.xlabel("Learning Rate")
plt.ylabel("Metric Value")
plt.title("Learning Rate Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("learning_rate_comparison_from_csv.png", dpi=300)
plt.show()

