import matplotlib.pyplot as plt

# FedHealth-XAI project models
models = [
    "Centralized\nML",
    "Local MLP\nModels",
    "Standard\nFederated Learning",
    "FL + Differential\nPrivacy",
    "FedHealth-XAI"
]

# Performance values (%)
accuracy = [
    75.32,   # Centralized ML
    46.00,   # Local MLP Models
    87.01,   # Standard Federated Learning
    57.14,   # FL + Differential Privacy
    46.75    # FedHealth-XAI
]

# Create figure
plt.figure(figsize=(14, 8))

bars = plt.bar(
    models,
    accuracy,
    width=0.62
)

# Title
plt.title(
    "FedHealth-XAI Model Performance Comparison",
    fontsize=24,
    fontweight="bold",
    pad=20
)

# Axis labels
plt.xlabel(
    "Model",
    fontsize=16
)

plt.ylabel(
    "Accuracy (%)",
    fontsize=16
)

# Y-axis
plt.ylim(0, 100)
plt.yticks(range(0, 101, 20), fontsize=13)

# X-axis
plt.xticks(fontsize=13)

# Display percentage above every bar
for bar, value in zip(bars, accuracy):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 2,
        f"{value:.2f}%",
        ha="center",
        va="bottom",
        fontsize=14,
        fontweight="bold"
    )

# Horizontal grid
plt.grid(
    axis="y",
    linestyle="-",
    alpha=0.2
)

plt.tight_layout()

# Save graph
plt.savefig(
    "results/graphs/fedhealth_xai_model_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# Hospital-wise performance comparison

import matplotlib.pyplot as plt
import numpy as np

hospitals = ["Hospital A", "Hospital B", "Hospital C"]

accuracy = [67.21, 83.61, 80.33]
precision = [100.00, 82.14, 72.22]
recall = [28.57, 82.14, 92.86]
f1_score = [44.44, 82.14, 81.25]

x = np.arange(len(hospitals))
width = 0.20

plt.figure(figsize=(14, 8))

bars1 = plt.bar(x - 1.5*width, accuracy, width, label="Accuracy")
bars2 = plt.bar(x - 0.5*width, precision, width, label="Precision")
bars3 = plt.bar(x + 0.5*width, recall, width, label="Recall")
bars4 = plt.bar(x + 1.5*width, f1_score, width, label="F1-Score")

plt.title(
    "Hospital-wise Model Performance Comparison",
    fontsize=24,
    fontweight="bold",
    pad=20
)

plt.xlabel("Hospital", fontsize=16)
plt.ylabel("Performance (%)", fontsize=16)

plt.xticks(x, hospitals, fontsize=14)

plt.ylim(0, 110)
plt.yticks(range(0, 101, 20), fontsize=13)

all_bars = [
    (bars1, accuracy),
    (bars2, precision),
    (bars3, recall),
    (bars4, f1_score)
]

for bars, values in all_bars:
    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.5,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold"
        )

plt.grid(axis="y", linestyle="-", alpha=0.2)

plt.legend(
    fontsize=13,
    loc="upper center",
    ncol=4
)

plt.tight_layout()

plt.savefig(
    "results/graphs/hospital_performance_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()