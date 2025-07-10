import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# Set the backend to Agg for non-interactive use
matplotlib.use('Agg')

# Example data for memory profiling (replace with actual values from logs)
agents = ['MCTS', 'HierarchicalMCTS']
memory_usage = [119.7, 119.8]  # Replace with actual memory usage values (MiB)
cpu_time = [1.186, 5.464]  # Replace with actual CPU time values (ms)
line_time = [1.75742,7.14596]

# Create subplots
fig, axes = plt.subplots(1, 3, figsize=(12, 6))

# Memory usage bar chart
axes[0].bar(agents, memory_usage, color=['#4CAF50', '#FFC107'])
axes[0].set_title('Memory Usage Comparison')
axes[0].set_ylabel('Memory Usage (MiB)')
for i, v in enumerate(memory_usage):
    axes[0].text(i, v + 0.1, f'{v:.1f} MiB', ha='center', va='bottom')

# CPU time bar chart
axes[1].bar(agents, cpu_time, color=['#2196F3', '#F44336'])
axes[1].set_title('CPU Time Comparison')
axes[1].set_ylabel('CPU Time (ms)')
for i, v in enumerate(cpu_time):
    axes[1].text(i, v + 5, f'{v:.1f} ms', ha='center', va='bottom')

# Line profiling time bar chart
axes[2].bar(agents, line_time, color=['#9C27B0', '#FF9800'])
axes[2].set_title('Line Profiling Time Comparison')
axes[2].set_ylabel('Line Profiling Time (ms)')
for i, v in enumerate(line_time):
    axes[2].text(i, v + 0.1, f'{v:.5f} ms', ha='center', va='bottom')
# Adjust layout and save plot
plt.tight_layout()
plt.savefig('profiling_comparison.png')  # Save the plot as an image