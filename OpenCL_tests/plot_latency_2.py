import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
data = pd.read_csv('latency_results.csv')

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(data['Index'], data['Latency'], marker='o')
plt.xlabel('Index')
plt.ylabel('Latency (seconds)')
plt.title('Memory Access Latency per Byte')
plt.grid(True)

# Save the plot
plt.savefig('latency_plot_2.png')
print('Plot saved as latency_plot.png')
