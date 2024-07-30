import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
data = pd.read_csv('latency_results.csv')

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(data['Stride'], data['Latency'], marker='o')
plt.xlabel('Stride')
plt.ylabel('Latency (seconds)')
plt.title('Memory Access Latency vs. Stride')
plt.grid(True)
plt.xscale('log')
plt.yscale('log')

# Save the plot
plt.savefig('latency_plot.png')
print('Plot saved as latency_plot.png')
