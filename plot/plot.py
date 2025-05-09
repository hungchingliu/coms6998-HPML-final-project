import matplotlib.pyplot as plt

# Data
methods = ['Full Training', 'LoRA Rank-1', 'LoRA Rank-16', 'LoRA Rank-64']
runtime = [2.56, 1.35, 1.89, 1.38]  # in seconds
memory = [20.26, 3.50, 3.54, 3.68]  # in GB

runtime_colors = ['skyblue', 'navy', 'navy', 'navy']
memory_colors = ['skyblue', 'navy', 'navy', 'navy']

# Plot 1: Runtime per Epoch
plt.figure(figsize=(6, 4))
bars = plt.bar(methods, runtime, color=runtime_colors)
plt.title('Runtime per Epoch')
plt.ylabel('Seconds')
plt.xticks(rotation=15)
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.05, f'{height:.2f}s', ha='center', va='bottom')
plt.tight_layout()
plt.savefig("runtime_per_epoch.png")
plt.show()

# Plot 2: Peak Memory Usage
plt.figure(figsize=(6, 4))
bars = plt.bar(methods, memory, color=memory_colors)
plt.title('Peak Memory Usage')
plt.ylabel('Memory (GB)')
plt.xticks(rotation=15)
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.1, f'{height}GB', ha='center', va='bottom')
plt.tight_layout()
plt.savefig("peak_memory_usage.png")
plt.show()
