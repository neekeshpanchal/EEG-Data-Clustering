import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
import time

start_time = time.time()

# Step 1: Load EEG data from eeg_data.csv (Optimization: Specify data types if applicable)
eeg_data = pd.read_csv("eeg_data.csv")

# Step 2: Data Normalization
scaler = StandardScaler()
normalized_eeg_data = scaler.fit_transform(eeg_data)

# Step 3: Feature Extraction - Find time point with highest activity for each electrode
highest_activity_time = np.argmax(np.abs(normalized_eeg_data), axis=0)

# Step 4: Clustering - Identify brain areas with similar activity during their highest activity time
n_clusters = 32  # Reduce the number of clusters for faster computation
kmeans = MiniBatchKMeans(n_clusters=n_clusters, n_init=3, random_state=42)
clusters = kmeans.fit_predict(normalized_eeg_data)

# Get the number of unique clusters found by the MiniBatchKMeans algorithm
n_unique_clusters = len(np.unique(clusters))

# Replace electrode_names with the appropriate labels for each electrode
electrode_names = ['Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7',
                  'PO3', 'O1', 'Oz', 'Pz', 'Fp2', 'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz',
                  'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2']

# Check if the number of unique clusters is less than n_clusters
if n_unique_clusters < n_clusters:
    # Handle mismatch by appending additional cluster labels and corresponding electrode names
    missing_clusters = n_clusters - n_unique_clusters
    additional_clusters = np.arange(n_unique_clusters, n_unique_clusters + missing_clusters)
    clusters = np.concatenate((clusters, additional_clusters))
    electrode_names = np.concatenate((electrode_names, [f"Unknown_{i}" for i in additional_clusters]))

# Check if electrode_names has the same length as the number of unique clusters
if len(electrode_names) != n_unique_clusters:
    raise ValueError("Electrode names and clusters must have the same length.")

# Create a DataFrame to store the clustering results
clustering_results = pd.DataFrame({'Electrode': electrode_names[:n_unique_clusters], 'Cluster': clusters[:n_unique_clusters]})

# Visualization using a bar plot and scatterplot side by side
plt.figure(figsize=(16, 6))

# Plot the bar graph
plt.subplot(1, 2, 1)
sns.countplot(data=clustering_results, x='Cluster', palette='viridis')
plt.xlabel('Cluster')
plt.ylabel('Number of Electrodes')
plt.title('Distribution of Electrodes in Clusters')

# Plot the scatterplot
plt.subplot(1, 2, 2)
sns.scatterplot(data=clustering_results, x='Cluster', y='Electrode', s=100, palette='viridis', hue='Cluster')
plt.xlabel('Cluster')
plt.ylabel('Electrode')
plt.title('Clusters and Electrodes with Highest Activity')
plt.legend(loc='upper right', title='Cluster')
for i in range(len(clustering_results)):
    plt.annotate(clustering_results['Electrode'][i], (clustering_results['Cluster'][i], clustering_results['Electrode'][i]),
                 textcoords="offset points", xytext=(0, 5), ha='center')

plt.tight_layout()
plt.show()

end_time = time.time()
execution_time = end_time - start_time
print("Execution Time:", execution_time, "seconds")
