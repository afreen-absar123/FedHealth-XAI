import matplotlib.pyplot as plt

# Model names
models = [
    "Centralized\nML",
    "Local MLP\nModels",
    "Standard\nFederated Learning",
    "FL + Differential\nPrivacy",
    "FedHealth-XAI\n(Adaptive Aggregation)"
]

# Performance values
performance = [75, 46, 52, 48, 50]

# Create the graph
plt.figure(figsize=(12, 7))

bars = plt.bar(
    models,
    performance,
    color="#1976B5",
    width=0.78
)

# Graph title
plt.title(
    "Model Performance Comparison",
    fontsize=22,
    fontweight="bold",
    pad=20
)

# Y-axis label
plt.ylabel(
    "Accuracy (%)",
    fontsize=15,
    fontweight="bold"
)

# Y-axis range
plt.ylim(0, 100)

# Y-axis ticks
plt.yticks(
    range(0, 101, 20),
    fontsize=12
)

# X-axis labels
plt.xticks(
    fontsize=12
)

# Add percentage values above each bar
for bar, value in zip(bars, performance):

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 2,
        f"{value}%",
        ha="center",
        va="bottom",
        fontsize=15,
        fontweight="bold"
    )

# Light horizontal grid
plt.grid(
    axis="y",
    linestyle="-",
    alpha=0.15
)

# Keep the graph clean
plt.gca().spines["top"].set_visible(True)
plt.gca().spines["right"].set_visible(True)

# Adjust spacing
plt.tight_layout()

# Save the graph
plt.savefig(
    "results/graphs/model_performance_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

# Display the graph
plt.show()