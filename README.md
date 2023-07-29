# EEG-Data-Clustering
EEG Clustering: found the time point with the highest activity for each electrode, clustered the data based on time points, visualized the clustered brain areas on a head map. Combination of data preprocessing, feature extraction, clustering, and data visualization tasks to gain insights into the brain activity patterns.

The code analyzes EEG data to identify brain areas with similar activity during their highest activity time. It follows these steps:
  
          1. Load EEG data from a CSV file into a pandas DataFrame.
          2. Normalize the data to ensure consistency in scaling across electrodes.
          3. Find the time point with the highest activity for each electrode.
          4. Use K-Means clustering to group electrodes based on their highest activity time.
          5. Visualize the clustered brain areas on a head map, indicating electrode names and their respective clusters.


Created for research and training purposes.
https://www.kaggle.com/datasets/samnikolas/eeg-dataset?resource=download (Data Source)
