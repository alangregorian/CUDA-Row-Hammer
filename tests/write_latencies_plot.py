import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
data = pd.read_csv('write_latencies.csv')

# Calculate the average latency
average_latency = data['Time'].mean()

# Plot the latencies
plt.figure(figsize=(15, 7))
plt.plot(data['Iteration'], data['Time'], label='Write Latency')
plt.axhline(y=average_latency, color='r', linestyle='--', label=f'Average Latency: {average_latency:.2f}')
plt.xlabel('Iteration')
plt.ylabel('Time (clock cycles)')
plt.title('Write Latencies to Global Memory')
plt.legend()
plt.grid(True)
plt.savefig('write_latency_plot.png')
