import pandas as pd
import matplotlib.pyplot as plt
import mplcursors  # Import the mplcursors library

# Load EEG data from CSV file
data = pd.read_csv("eeg_data.csv")

# EEG electrode names (columns used)
electrode_names = ['Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7',
                  'PO3', 'O1', 'Oz', 'Pz', 'Fp2', 'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz',
                  'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2']

# Assuming data is sampled at a fixed frequency (e.g., 1 Hz)
sampling_rate = 60  # 60 samples per minute

# Create time points based on the number of data points and sampling rate (in minutes)
time_points = pd.Index(range(len(data))) / sampling_rate

# Create a subplot grid based on the number of electrodes
num_electrodes = len(electrode_names)
num_cols = 4  # Number of columns in the subplot grid (adjust as needed)
num_rows = (num_electrodes + num_cols - 1) // num_cols

# Define the locations of the EEG electrodes (Replace with your own locations)
electrode_locations = {
    'Fp1': 'Frontal (Frontopolar)',
    'AF3': 'Frontal (Anterior Frontal)',
    'F3': 'Frontal (Left Frontal)',
    'F7': 'Frontal (Left Frontal)',
    'FC5': 'Frontal (Left Frontal)',
    'FC1': 'Frontal (Left Frontal)',
    'C3': 'Central (Left Central)',
    'T7': 'Temporal (Left Temporal)',
    'CP5': 'Parietal (Left Parietal)',
    'CP1': 'Parietal (Left Parietal)',
    'P3': 'Parietal (Left Parietal)',
    'P7': 'Parietal (Left Parietal)',
    'PO3': 'Occipital (Left Occipital)',
    'O1': 'Occipital (Left Occipital)',
    'Oz': 'Occipital (Occipital)',
    'Pz': 'Parietal (Parietal)',
    'Fp2': 'Frontal (Frontopolar)',
    'AF4': 'Frontal (Anterior Frontal)',
    'Fz': 'Frontal (Frontal)',
    'F4': 'Frontal (Right Frontal)',
    'F8': 'Frontal (Right Frontal)',
    'FC6': 'Frontal (Right Frontal)',
    'FC2': 'Frontal (Right Frontal)',
    'Cz': 'Central (Central)',
    'C4': 'Central (Right Central)',
    'T8': 'Temporal (Right Temporal)',
    'CP6': 'Parietal (Right Parietal)',
    'CP2': 'Parietal (Right Parietal)',
    'P4': 'Parietal (Right Parietal)',
    'P8': 'Parietal (Right Parietal)',
    'PO4': 'Occipital (Right Occipital)',
    'O2': 'Occipital (Right Occipital)',
}

# Plot EEG data for each electrode in a single window
plt.figure(figsize=(15, 10))
for i, electrode in enumerate(electrode_names):
    plt.subplot(num_rows, num_cols, i + 1)
    line = plt.plot(time_points, data[electrode])[0]
    plt.title(f"Electrode {electrode}")
    plt.xlabel("Time (minutes)")
    plt.ylabel("Voltage\n(microvolts)", fontsize=8)  # Adjust the font size here
    plt.grid(True)
    
    # Set an even smaller font size for y-axis tick labels
    plt.tick_params(axis='y', labelsize=8)
    
    # Function to provide tooltip information
    def tooltip_handler(sel):
        label = sel.artist.get_label()
        sel.annotation.set_text(f"Location: {electrode_locations[label]}")
        sel.annotation.get_bbox_patch().set(fc="white", alpha=0.9)  # Customize tooltip style

    # Add tooltip with electrode location using mplcursors
    mplcursors.cursor(hover=True).connect("add", tooltip_handler)

    line.set_label(electrode)  # Set label for the line

# Adjust subplot layout and spacing
plt.tight_layout()

# Show the combined plot
plt.show()
