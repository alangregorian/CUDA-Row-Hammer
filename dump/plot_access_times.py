import matplotlib.pyplot as plt
import numpy as np

# Initialize lists for strides and times
strides = []
times = []

# Read the data from the file
with open('access_times.txt', 'r') as file:
    for line in file:
        stride, time = line.strip().split()
        strides.append(int(stride))
        times.append(int(time))

# Convert lists to numpy arrays for easier manipulation
strides = np.array(strides)
times = np.array(times)

# Find unique strides and calculate average times for each stride
unique_strides = sorted(set(strides))
average_times = []

for stride in unique_strides:
    stride_times = times[strides == stride]
    average_times.append(np.mean(stride_times))

# Create a bar graph
plt.figure(figsize=(12, 6))
plt.bar(range(len(unique_strides)), average_times, tick_label=unique_strides)
plt.xlabel('Stride Index')
plt.ylabel('Access Time (clock cycles)')
plt.title('Memory Access Time for Different Strides')
plt.xticks(rotation=45)
plt.grid(axis='y')

# Save the plot as an image file
plt.savefig('access_times_bar_plot.png')
