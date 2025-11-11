import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np

# 1 = quantum wins (yellow-orange), 0 = classical wins (blue)
round_results = [
    1,1,1,0,1,1,0,1,1,0,
    1,1,0,1,0,1,1,1,1,1,
    1,1,1,1,1,1,1,1,1,0,
    1,1,1,1,1,1,1,1,1,0,
    0,1,1,1,0,1,1,1,0,1,
    1,1,1,1,1,1,1,1,0,1,
    1,1,1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,0,1,0,1,
    1,1,1,1,1,0,1,1,1,1,
    1,1,1,1,1,1,1,1,1,1
]

# Convert to 10x10 grid
grid = np.array(round_results).reshape((10,10))

# Color map: 0=blue (classical), 1=yellow-orange (quantum)
cmap = mcolors.ListedColormap(['#add8e6', '#F9B234'])

plt.figure(figsize=(8,8))
plt.imshow(grid, cmap=cmap, vmin=0, vmax=1)

# Add grid lines
for i in range(11):
    plt.axhline(i-0.5, color='k', linewidth=0.5)
    plt.axvline(i-0.5, color='k', linewidth=0.5)

# Label each block
for i in range(10):
    for j in range(10):
        block_num = i*10 + j + 1
        plt.text(j, i, f'{block_num}', ha='center', va='center', color='black', fontsize=8)

# Legend (placed outside the grid so it doesn't overlap)
quantum_patch = mpatches.Patch(color='#F9B234', label='Quantum Algorithm')
classical_patch = mpatches.Patch(color='blue', label='Classical Algorithm')
plt.legend(handles=[classical_patch, quantum_patch], loc='upper left', bbox_to_anchor=(1.02, 1))

plt.title("Quantum vs Classical Race: 100 Blocks")
plt.axis('off')
plt.tight_layout()
plt.show()
