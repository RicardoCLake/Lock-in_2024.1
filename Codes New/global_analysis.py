import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import warnings

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Datasets using strictly the '../Mesures/' folder
DATA_FILES = [
    '../Mesures/data01.csv.old', 
    '../Mesures/data02.csv.old', 
    '../Mesures/data03.csv',
    '../Mesures/data_test_iphone.csv', 
    '../Mesures/data_test_vitor.csv'
]

def load_single_dataset(file_path):
    if not os.path.exists(file_path):
        print(f"\n[ERROR] File {file_path} not found. Skipping...")
        return None, None
        
    print(f"\n{'='*60}")
    print(f"--- Loading Data: {file_path} ---")
    df = pd.read_csv(file_path, index_col=0)
    
    # Clean the device ID (convert "5" or 5 to "I")
    if 'device_id' in df.columns:
        df['device_id'] = df['device_id'].replace({'5': 'I', 5: 'I'})
        
    dataset_name = os.path.basename(file_path).replace('.csv', '')
    print(f"Data Shape: {df.shape}")
    return df, dataset_name

def feature_extraction(df):
    num_samples = len(df)
    num_features = len(df.columns)
    
    # Robust Regex to strictly catch MAC addresses (e.g., 00:1A:2B:3C:4D:5E)
    mac_address_pattern = re.compile(r'([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})')
    
    features_mac = [col for col in df.columns if mac_address_pattern.search(str(col))]
    features_labels = [col for col in df.columns if col not in features_mac]

    print(f"Total Samples: {num_samples} | Total Features: {num_features}")
    print(f"Metadata Columns: {len(features_labels)} | Total MAC Address Columns: {len(features_mac)}\n")
    
    return features_labels, features_mac

def get_filtered_macs(df_macs):
    # Load the relation file to get the specific 409 MACs
    if not os.path.exists("../Mesures/mac_name_relation.csv"):
        print("[WARNING] ../Mesures/mac_name_relation.csv not found in the current directory. Cannot filter target MACs.")
        return []
        
    relation = pd.read_csv("../Mesures/mac_name_relation.csv", index_col=0)
    target_ssids = ["Guest-CentraleSupelec", "eduroam", 'stop&go', 'CD91', 'fabrique2024']
    
    good_aps = relation[relation['ap_name'].isin(target_ssids)]["ap_mac"].to_list()
    
    # Only keep the target MACs that are actually present in the current dataframe
    filtered_macs = [mac for mac in good_aps if mac in df_macs]
    print(f"Target Filtered MAC Addresses found in this dataset: {len(filtered_macs)}\n")
    
    return filtered_macs

def analyze_nan_percentages(df, features_mac, dataset_name, analysis_type="all"):
    print(f"--- NaN Values Analysis per Device ({analysis_type.upper()} MACs) ---")
    
    if not features_mac:
        print("No MAC addresses provided for this analysis. Skipping.")
        return

    devices = sorted([str(d) for d in df['device_id'].dropna().unique()])
    percentages_per_device = []
    
    if not devices:
        print("No devices found in this dataset.")
        return

    for dev in devices:
        device_data = df[df['device_id'] == dev][features_mac]
        
        if not device_data.empty and device_data.size > 0:
            total_cells = device_data.size
            total_nans = device_data.isna().sum().sum()
            pct = (total_nans / total_cells) * 100
        else:
            pct = 0
            
        percentages_per_device.append(pct)
        print(f"Device {dev}: {pct:.2f}% missing values")

    # --- Plotting ---
    plt.figure(figsize=(10, 6))
    
    # Change color scheme slightly depending on the analysis type
    if analysis_type == "filtered":
        colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(devices)))
        title_suffix = "Filtered Target MACs (5 SSIDs)"
    else:
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(devices)))
        title_suffix = "All MACs"

    bars = plt.bar(devices, percentages_per_device, color=colors, edgecolor='black', alpha=0.8)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.1f}%', ha='center', va='bottom', fontweight='bold')

    plt.xlabel('Device ID', fontsize=12, fontweight='bold')
    plt.ylabel('Percentage of NaN Values (%)', fontsize=12, fontweight='bold')
    plt.title(f'Missing Values (NaN) per Device - {dataset_name} [{title_suffix}]', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Set reasonable Y limits
    max_pct = max(percentages_per_device) if percentages_per_device else 0
    min_pct = min(percentages_per_device) if percentages_per_device else 0
    plt.ylim(top=min(100, max_pct + 10), bottom=max(0, min_pct - 10))
    
    # Save the plot dynamically named after the dataset and analysis type
    plot_path = os.path.join(OUTPUT_DIR, f'nan_comparison_{dataset_name}_{analysis_type}_macs.png')
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"[SAVED] NaN visualization written to: {plot_path}\n")

def analyze_samples_per_room(df):
    print("--- Samples Analysis per Room and Device ---")
    
    rooms = sorted([r for r in df['room'].unique() if pd.notna(r)])
    print(f"Quantity of unique rooms: {len(rooms)}\n")
    
    # Clean Table Header
    print(f"{'Room':<10} | {'Devices':<30} | {'Samples per device'}")
    print("-" * 80)
    
    for room in rooms:
        room_data = df[df['room'] == room]
        
        devices_in_room = sorted([str(d) for d in room_data['device_id'].dropna().unique()])
        num_devices = len(devices_in_room)
        
        counts = room_data['device_id'].value_counts()
        
        samples_per_device = []
        for dev in devices_in_room:
            # Using int() to remove np.int64 wrapping from the print output
            samples_per_device.append(int(counts.get(dev, 0)))
            
        devices_str = f"{devices_in_room} ({num_devices})"
        
        print(f"{room:<10} | {devices_str:<30} | {samples_per_device}")
        
    print("-" * 80 + "\n")

# --- MAIN RUN ---
if __name__ == "__main__":
    
    all_loaded_datasets = []
    
    # 1. Load all 5 individual datasets into memory
    for file_path in DATA_FILES:
        df_single, dataset_name = load_single_dataset(file_path)
        if df_single is not None:
            all_loaded_datasets.append((df_single, dataset_name))

    # 2. Create the 6th dataset (merge 03, test_vitor, and test_iphone)
    print(f"\n{'='*60}")
    print("--- Creating 6th Dataset: Merging data03, test_vitor, and test_iphone ---")
    
    dfs_to_merge = []
    for df, name in all_loaded_datasets:
        if name in ['data03', 'data_test_vitor', 'data_test_iphone']:
            dfs_to_merge.append(df)
            
    if dfs_to_merge:
        merged_6th_df = pd.concat(dfs_to_merge, ignore_index=True)
        print(f"Merged 6th Dataset Shape: {merged_6th_df.shape}")
        all_loaded_datasets.append((merged_6th_df, 'merged_03_vitor_iphone'))
    else:
        print("Warning: Could not create the 6th dataset. Target files not found.")

    # 3. Run the analysis pipeline on ALL 6 datasets
    for df, dataset_name in all_loaded_datasets:
        print(f"\n{'#'*60}")
        print(f"### EXECUTING ANALYSIS FOR: {dataset_name} ###")
        print(f"{'#'*60}")
        
        # Extract feature names (All MACs)
        labels, all_macs = feature_extraction(df)
        
        # Filter for the Target MACs
        filtered_macs = get_filtered_macs(all_macs)
        
        # Analyze NaNs for ALL MACs
        analyze_nan_percentages(df, all_macs, dataset_name, analysis_type="all")
        
        # Analyze NaNs for FILTERED MACs (Target 409)
        analyze_nan_percentages(df, filtered_macs, dataset_name, analysis_type="filtered")
        
        # Analyze room distribution
        analyze_samples_per_room(df)
            
    print("\n--- Pipeline Execution Finished Successfully ---")