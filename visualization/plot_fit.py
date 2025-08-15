import pandas as pd
import matplotlib.pyplot as plt
import sys
# Read the CSV file (change filename as needed)
filename = sys.argv[1]
df = pd.read_csv(filename)

# Plot step vs loss
plt.figure(figsize=(8, 5))
plt.plot(df["step"], df["loss"], marker='o', linestyle='-', color='b', label='Loss')

plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Step vs. Loss")
plt.grid(True)
plt.legend()
plt.tight_layout()

# Show the plot
plt.show()
