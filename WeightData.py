import json

with open("weights.json") as f:
    weights = json.load(f)

# Get indices sorted by absolute value (smallest first)
sorted_indices = sorted(range(len(weights)), key=lambda i: abs(weights[i]))

# Print smallest 2
print("Two smallest weights by absolute value:")
for i in sorted_indices[:5]:
    print(f"Index {i}: {weights[i]}")