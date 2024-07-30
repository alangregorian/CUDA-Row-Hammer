import pandas as pd
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv("results.csv")

# Plot data
plt.scatter(data['Index'], data['ZeroTime'], label='Zero Time', color='r')
plt.scatter(data['Index'], data['IndexTime'], label=f'Index {N*500} Time', color='b')

# Labeling the axes
plt.xlabel('Index')
plt.ylabel('Latency (clock cycles)')
plt.title('Latency vs Index')

# Adding a legend
plt.legend()

# Save plot to file
plt.savefig("latency_plot.png")
