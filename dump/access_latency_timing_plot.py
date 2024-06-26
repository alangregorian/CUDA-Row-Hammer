import matplotlib.pyplot as plt
import pandas as pd

# Load the data from the CSV file
data = pd.read_csv('access_latency_timing.csv')

# Create a scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(data['BytesFromZero'], data['TimeZero'], color='blue', label='Access Time to Zero Index')
plt.scatter(data['BytesFromZero'], data['TimeN100'], color='red', label='Access Time to N*100 Bytes')

# Set the labels and title
plt.xlabel('Index (Bytes)')
plt.ylabel('Latency (microseconds)')
plt.title('Global Memory Access Latency')
plt.legend()

# Save the plot to a file
plt.grid(True)
plt.savefig('access_latency_plot.png')
