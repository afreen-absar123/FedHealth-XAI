import matplotlib.pyplot as plt

# Actual values from model_comparison.csv
models = [
    "Centralized",
    "Adaptive Aggregation",
    "FedHealth-XAI"
]

accuracy = [
    75.32,
    87.01,
    46.75
]

# Create graph
plt.figure(figsize=(12, 7))

bars = plt.bar(
    models,
    accuracy,
    width=0.65
)

# Title
plt.title(
    "Model Performance Comparison",
    fontsize=22,
    fontweight="bold",
    pad=20
)

# Axis labels
plt.xlabel(
    "Model",
    fontsize=15
)

plt.ylabel(
    "Accuracy (%)",
    fontsize=15
)

# Y-axis
plt.ylim(0, 100)
plt.yticks(range(0, 101, 20), fontsize=12)

# X-axis
plt.xticks(fontsize=13)

# Percentage labels
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

# Grid
plt.grid(
    axis="y",
    linestyle="-",
    alpha=0.2
)

plt.tight_layout()

# Save inside your existing results folder
plt.savefig(
    "results/graphs/correct_model_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()