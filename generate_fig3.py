import matplotlib.pyplot as plt

# --- Data ---
modules = ['Trie Autocomplete', 'D–L Corrector']
times = [0.11, 4330.63]

# --- Plot ---
plt.figure(figsize=(6, 4))
plt.bar(modules, times, color=['#4c72b0', '#55a868'])
plt.yscale('log')  # Use logarithmic scale for readability
plt.ylabel('Average Time (ms, log scale)')
plt.title('Performance Comparison: Trie vs D–L Corrector')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# --- Annotate bars with their values ---
for i, v in enumerate(times):
    plt.text(i, v * 1.5, f"{v:.2f}", ha='center', fontsize=9)

# --- Save image to file ---
plt.tight_layout()
plt.savefig('fig3.png', dpi=300)
plt.show()

print("✅ Figure saved as fig3.png in your project folder!")
