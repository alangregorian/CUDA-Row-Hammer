import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
data = pd.read_csv('latencies.csv')

# Calculate the differences between consecutive latency measurements
data['Latency_Diff'] = data['Latency'].diff().fillna(0)

# Remove instances where the difference is zero
non_zero_diff_data = data[data['Latency_Diff'] != 0]

# Plot the histogram of latency differences
plt.figure(figsize=(10, 6))
plt.hist(non_zero_diff_data['Latency_Diff'], bins=100, edgecolor='black')
plt.xlabel('Latency Difference (clock cycles)')
plt.ylabel('Number of Occurrences')
plt.title('Distribution of Differences Between Consecutive Latency Measurements (Non-zero Differences)')
plt.grid(True)
plt.savefig('latency_differences_histogram_non_zero.png')
