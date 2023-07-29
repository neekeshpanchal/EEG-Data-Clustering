import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import time

start_time = time.time()
# Step 1: Load EEG data from eeg_data.csv
eeg_data = pd.read_csv("eeg_data.csv")

# Step 2: Data Normalization
scaler = StandardScaler()
normalized_eeg_data = scaler.fit_transform(eeg_data)

# Step 3: Feature Extraction - Find time point with highest activity for each electrode
# Transpose the normalized_eeg_data to ensure the 'argmax' function is applied along the correct axis
highest_activity_time = np.argmax(np.abs(normalized_eeg_data), axis=1)

# Step 4: Clustering - Identify brain areas with similar activity during their highest activity time
n_clusters = len(highest_activity_time)  # Number of clusters set to the number of electrodes
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(normalized_eeg_data[:, highest_activity_time])

# Step 5: Visualization - Visualize clustered brain areas on a head map
# Replace electrode_names with the appropriate labels for each electrode
electrode_names = ['Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7',
                  'PO3', 'O1', 'Oz', 'Pz', 'Fp2', 'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz',
                  'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2']

# Create a DataFrame to store the clustering results
clustering_results = pd.DataFrame({'Electrode': electrode_names, 'Cluster': clusters})

# Visualization using a head map
plt.figure(figsize=(10, 6))
sns.scatterplot(data=clustering_results, x='Electrode', y='Cluster', s=100, palette='viridis', hue='Cluster')
plt.xlabel('Electrode')
plt.ylabel('Cluster')
plt.title('Brain Area Clustering during Highest Activity Time')
plt.show()
end_time = time.time()
execution_time = end_time - start_time
print("Execution Time:", execution_time, "seconds")