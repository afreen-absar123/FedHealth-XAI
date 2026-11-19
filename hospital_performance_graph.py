import matplotlib.pyplot as plt
import numpy as np
import os


# ==========================================
# Hospital names
# ==========================================

hospitals = [
    "Hospital A",
    "Hospital B",
    "Hospital C"
]


# ==========================================
# Performance metrics
# Values from the latest model comparison
# ==========================================

accuracy = [
    57.14,
    77.92,
    83.12
]

precision = [
    100.00,
    81.58,
    81.82
]

recall = [
    19.51,
    75.61,
    87.80
]

f1_score = [
    32.65,
    78.48,
    84.71
]


# ==========================================
# Create output folders
# ==========================================

os.makedirs(
    "static/graphs",
    exist_ok=True
)

os.makedirs(
    "results/graphs",
    exist_ok=True
)


# ==========================================
# Position of each hospital
# ==========================================

x = np.arange(len(hospitals))

# Width of each bar
width = 0.18


# ==========================================
# Create figure
# ==========================================

plt.figure(
    figsize=(14, 8)
)


# ==========================================
# Create bars
# ==========================================

bars1 = plt.bar(
    x - 1.5 * width,
    accuracy,
    width,
    label="Accuracy"
)

bars2 = plt.bar(
    x - 0.5 * width,
    precision,
    width,
    label="Precision"
)

bars3 = plt.bar(
    x + 0.5 * width,
    recall,
    width,
    label="Recall"
)

bars4 = plt.bar(
    x + 1.5 * width,
    f1_score,
    width,
    label="F1 Score"
)


# ==========================================
# Title
# ==========================================

plt.title(
    "FedHealth-XAI Hospital-wise Performance Metrics",
    fontsize=24,
    fontweight="bold",
    pad=20
)


# ==========================================
# X-axis
# ==========================================

plt.xlabel(
    "Simulated Hospital",
    fontsize=16,
    fontweight="bold"
)

plt.xticks(
    x,
    hospitals,
    fontsize=14
)


# ==========================================
# Y-axis
# ==========================================

plt.ylabel(
    "Performance (%)",
    fontsize=16,
    fontweight="bold"
)

plt.yticks(
    fontsize=13
)

plt.ylim(
    0,
    110
)


# ==========================================
# Add percentage labels
# ==========================================

def add_labels(bars):

    for bar in bars:

        height = bar.get_height()

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 1.5,
            f"{height:.2f}%",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold"
        )


add_labels(bars1)
add_labels(bars2)
add_labels(bars3)
add_labels(bars4)


# ==========================================
# Legend
# ==========================================

plt.legend(
    fontsize=13,
    loc="upper right",
    ncol=2
)


# ==========================================
# Horizontal grid
# ==========================================

plt.grid(
    axis="y",
    linestyle="-",
    alpha=0.18
)


# ==========================================
# Clean graph borders
# ==========================================

ax = plt.gca()

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)


# ==========================================
# Layout
# ==========================================

plt.tight_layout()


# ==========================================
# Save for dashboard
# ==========================================

plt.savefig(
    "static/graphs/hospital_wise_performance.png",
    dpi=300,
    bbox_inches="tight"
)


# ==========================================
# Save copy for results
# ==========================================

plt.savefig(
    "results/graphs/hospital_wise_performance.png",
    dpi=300,
    bbox_inches="tight"
)


# ==========================================
# Close figure
# ==========================================

plt.close()


# ==========================================
# Confirmation
# ==========================================

print()
print("==========================================")
print("HOSPITAL-WISE PERFORMANCE GRAPH CREATED")
print("==========================================")

print("Hospital A:")
print("  Accuracy :", "57.14%")
print("  Precision:", "100.00%")
print("  Recall   :", "19.51%")
print("  F1 Score :", "32.65%")

print()

print("Hospital B:")
print("  Accuracy :", "77.92%")
print("  Precision:", "81.58%")
print("  Recall   :", "75.61%")
print("  F1 Score :", "78.48%")

print()

print("Hospital C:")
print("  Accuracy :", "83.12%")
print("  Precision:", "81.82%")
print("  Recall   :", "87.80%")
print("  F1 Score :", "84.71%")

print()
print("Dashboard graph saved to:")
print("static/graphs/hospital_wise_performance.png")

print()