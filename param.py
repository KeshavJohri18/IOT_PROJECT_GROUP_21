import matplotlib.pyplot as plt
import numpy as np

# --- APPLY STYLING ---
plt.style.use('seaborn-v0_8-whitegrid')

# --- DATA ---
experiments = [
    'Benchmark', 
    'Random\nSampling', 
    'Confidence\nThreshold', 
    'CNN\nModule', 
    'Deeper\nMLP'
]

# Estimated Parameter Counts (Millions)
# 1. Benchmark: Standard SGLATrack-DeiT*
# 2. Random Sampling: Benchmark MINUS the MLP (5.81 - 0.057 = ~5.75)
# 3. Confidence: Same architecture as Benchmark (5.81)
# 4. CNN: Estimated overhead of CNN weights (~6.10)
# 5. Deep MLP: Benchmark PLUS 2 extra hidden layers (~5.86)
param_counts = [5.81, 5.75, 5.81, 6.10, 5.86]

# --- CONSISTENT COLOR SCHEME ---
# Matches the AUC/FPS/Precision plots exactly
colors = ['#2ca02c', '#1f77b4', '#6baed6', '#d62728', '#d62728']

# --- PLOTTING ---
plt.figure(figsize=(10, 6))
bars = plt.bar(experiments, param_counts, color=colors, width=0.6, alpha=0.9)

plt.title('Model Complexity (Parameter Count)', fontsize=16, fontweight='bold', pad=15)
plt.ylabel('Parameters (Millions)', fontsize=12)
plt.xlabel('Method', fontsize=12, labelpad=10)

# Zoom in to show differences (Scale 5.5M to 6.2M)
plt.ylim(5.5, 6.2) 

# Benchmark Line
plt.axhline(y=5.81, color='green', linestyle='--', linewidth=1.5, alpha=0.6, label='Benchmark')

# Add Labels
for bar in bars:
    height = bar.get_height()
    plt.annotate(f'{height}M', 
                 xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 5), textcoords="offset points",
                 ha='center', va='bottom', fontsize=11, fontweight='bold', color='black')

plt.tight_layout()
plt.show()  