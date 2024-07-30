import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load the CSV file
data = pd.read_csv('latencies.csv')

# Perform KMeans clustering to identify potential clusters in the latency data
kmeans = KMeans(n_clusters=2)
data['Cluster'] = kmeans.fit_predict(data[['Latency']])

# Calculate the average latency for each cluster
cluster_averages = data.groupby('Cluster')['Latency'].mean()

# Plot the clustered data
plt.figure(figsize=(10, 6))
colors = ['red', 'blue']
for cluster in range(2):
    subset = data[data['Cluster'] == cluster]
    plt.hist(subset['Latency'], bins=100, edgecolor='black', color=colors[cluster], alpha=0.6, label=f'Cluster {cluster}')
    plt.axvline(cluster_averages[cluster], color=colors[cluster], linestyle='dashed', linewidth=2)
    # Add text annotation for the mean value
    plt.text(cluster_averages[cluster], plt.ylim()[1] * 0.9, f'{cluster_averages[cluster]:.2f}', color='black', ha='center')

plt.xlabel('Latency (clock cycles)')
plt.ylabel('Number of Occurrences')
plt.title('Distribution of Global Memory Access Latencies with Clusters and Averages')
plt.legend()
plt.grid(True)
plt.savefig('latency_clusters_with_averages.png')
plt.show()
