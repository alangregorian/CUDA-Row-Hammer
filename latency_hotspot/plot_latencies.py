import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
data = pd.read_csv('latencies.csv')

# Extract the initial and repeated latencies
initial_latencies = data['Initial Latency'].dropna()
# repeated_latencies = data['Repeated Latency'].dropna()

# Plot the histogram of initial latencies
plt.figure(figsize=(10, 6))
plt.hist(initial_latencies, bins=100, edgecolor='black')
plt.xlabel('Initial Latency (clock cycles)')
plt.ylabel('Number of Occurrences')
plt.title('Distribution of Initial Global Memory Access Latencies')
plt.grid(True)
plt.savefig('initial_latency_histogram.png')

# Plot the histogram of repeated latencies
# plt.figure(figsize=(10, 6))
# plt.hist(repeated_latencies, bins=100, edgecolor='black')
# plt.xlabel('Repeated Latency (clock cycles)')
# plt.ylabel('Number of Occurrences')
# plt.title('Distribution of Repeated Global Memory Access Latencies')
# plt.grid(True)
# plt.savefig('repeated_latency_histogram.png')

