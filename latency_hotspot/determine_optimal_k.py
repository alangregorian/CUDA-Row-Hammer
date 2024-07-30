import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load the CSV file
data = pd.read_csv('latencies.csv')

# Determine the optimal number of clusters using the Elbow Method
sum_of_squared_distances = []
K = range(1, 10)
for k in K:
    kmeans = KMeans(n_clusters=k)
    kmeans = kmeans.fit(data[['Latency']])
    sum_of_squared_distances.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(K, sum_of_squared_distances, 'bx-')
plt.xlabel('Number of clusters')
plt.ylabel('Sum of squared distances')
plt.title('Elbow Method For Optimal k')
plt.grid(True)
plt.savefig('elbow_method.png')

# Determine the optimal number of clusters using the Silhouette Score
silhouette_scores = []
for k in K[1:]:  # Silhouette score is not defined for k=1
    kmeans = KMeans(n_clusters=k)
    labels = kmeans.fit_predict(data[['Latency']])
    silhouette_scores.append(silhouette_score(data[['Latency']], labels))

plt.figure(figsize=(10, 6))
plt.plot(K[1:], silhouette_scores, 'bx-')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score For Optimal k')
plt.grid(True)
plt.savefig('silhouette_score.png')
