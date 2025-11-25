import matplotlib.pyplot as plt
import numpy as np

# --- APPLY STYLING ---
plt.style.use('seaborn-v0_8-whitegrid')

# --- COMMON DATA ---
experiments = [
    'Benchmark', 
    'Random\nSampling', 
    'Confidence\nThreshold', 
    'CNN\nModule', 
    'Deeper\nMLP'
]

# Color Scheme
colors = ['#2ca02c', '#1f77b4', '#6baed6', '#d62728', '#d62728']

# Helper function for labels
def add_bar_labels(bars):
    for bar in bars:
        height = bar.get_height()
        # Formatting: Integers for FPS, Percentages for others
        label = f'{height}%' if height > 40 else f'{height}'
        plt.annotate(label, 
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 5), textcoords="offset points",
                     ha='center', va='bottom', fontsize=11, fontweight='bold', color='black')

# --- FIGURE 1: AUC COMPARISON ---
auc_scores = [65.01, 64.21, 63.20, 54.91, 58.52]

plt.figure(figsize=(10, 6))
bars_auc = plt.bar(experiments, auc_scores, color=colors, width=0.6, alpha=0.9)

# UPDATED: Title changed to remove the word "Accuracy"
plt.title('AUC Performance Comparison', fontsize=16, fontweight='bold', pad=15)
plt.ylabel('AUC Score (%)', fontsize=12)
plt.xlabel('Method', fontsize=12, labelpad=10)
plt.ylim(50, 70) 
plt.axhline(y=65.01, color='green', linestyle='--', linewidth=1.5, alpha=0.6, label='Benchmark')

add_bar_labels(bars_auc)
plt.legend()
plt.tight_layout()
plt.show()


# --- FIGURE 2: PRECISION COMPARISON ---
precision_scores = [81.75, 81.87, 80.44, 73.07, 74.85]

plt.figure(figsize=(10, 6))
bars_prec = plt.bar(experiments, precision_scores, color=colors, width=0.6, alpha=0.9)

plt.title('Precision Comparison', fontsize=16, fontweight='bold', pad=15)
plt.ylabel('Precision (%)', fontsize=12)
plt.xlabel('Method', fontsize=12, labelpad=10)
plt.ylim(60, 90) 

# Green line lowered to 81.75 as requested
plt.axhline(y=81.75, color='green', linestyle='--', linewidth=1.5, alpha=0.6, label='Benchmark')

add_bar_labels(bars_prec)
plt.legend()
plt.tight_layout()
plt.show()


# --- FIGURE 3: FPS COMPARISON ---
local_fps = [27.84, 28.71, 28.71, 22.46, 21.02]

plt.figure(figsize=(10, 6))
bars_fps = plt.bar(experiments, local_fps, color=colors, width=0.6, alpha=0.9)

plt.title('Speed Comparison (Local Hardware)', fontsize=16, fontweight='bold', pad=15)
plt.ylabel('Frames Per Second (FPS)', fontsize=12)
plt.xlabel('Method', fontsize=12, labelpad=10)
plt.ylim(15, 35)
plt.axhline(y=27.84, color='green', linestyle='--', linewidth=1.5, alpha=0.6, label='Benchmark')

add_bar_labels(bars_fps)
plt.legend()
plt.tight_layout()
plt.show()