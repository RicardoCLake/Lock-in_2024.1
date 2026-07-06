import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import networkx as nx
from community import community_louvain # requires pip install python-louvain
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- STEP 1: LOAD AND MERGE DATA ---
def load_and_merge_data():
    print("--- Loading and Merging Data ---")
    files = [#'../Mesures/data01.csv', 
             #'../Mesures/data02.csv', 
             '../Mesures/data03.csv']
    dfs = []
    
    for file in files:
        if os.path.exists(file):
            print(f"Reading {file}...")
            # Assuming first column is an unnamed index, index_col=0 handles it
            df = pd.read_csv(file, index_col=0)
            dfs.append(df)
        else:
            print(f"Warning: {file} not found.")

    if not dfs:
        raise FileNotFoundError("No data files found. Check your paths.")

    # Merge all dataframes. Missing MACs across files will become NaNs
    merged_df = pd.concat(dfs, ignore_index=True)
    print(f"Initial Merged Data Shape: {merged_df.shape}")
    return merged_df

# --- STEP 2: APPLY YOUR MAC SELECTION & PREPROCESS ---
def preprocess_and_filter(data):
    print("\n--- Filtering Target MAC Addresses & Preprocessing ---")
    
    # 1. Load the relation file
    if not os.path.exists("../Mesures/mac_name_relation.csv"):
        raise FileNotFoundError("../Mesures/mac_name_relation.csv not found in the current working directory.")
        
    name_mac_relation = pd.read_csv("../Mesures/mac_name_relation.csv", index_col=0)
    
    # 2. Filter specific target SSIDs/AP names (Your exact selection logic)
    good_aps = name_mac_relation[name_mac_relation['ap_name'].isin(
        ["Guest-CentraleSupelec", "eduroam", 'stop&go', 'CD91', 'fabrique2024']
    )]["ap_mac"].to_list()
    
    # Target columns: The first 5 metadata columns + the good MAC addresses
    columns_to_maintain = good_aps + data.columns[:5].to_list()
    
    # Intersection prevents errors if some MACs from the relation file are missing in the data
    data = data[data.columns.intersection(columns_to_maintain)]
    print(f"Filtered Data Shape: {data.shape}")
    
    # Identify which columns are actually the MAC addresses now
    meta_cols = ['timestamp', 'room', 'device_id', 'door_status', 'room_part']
    mac_cols = [col for col in data.columns if col not in meta_cols]
    print(f"Number of target MAC addresses found and maintained: {len(mac_cols)}")
    
    # 3. Clean and normalize signal entries
    # Fill NaNs with (global minimum - 1) to represent an undetectable signal
    min_value = data[mac_cols].min().min()
    data[mac_cols] = data[mac_cols].fillna(min_value - 1)
    
    # Shift signals so the absolute minimum becomes 0
    data[mac_cols] = data[mac_cols] + (min_value - 1) * (-1)
    
    return data, mac_cols

# --- STRATEGY 1: MAC PREFIX CLUSTERING ---
def strategy_1_mac_prefix(mac_cols):
    print("\n--- Strategy 1: MAC Prefix Clustering ---")
    prefixes = {}
    for mac in mac_cols:
        # Group by the first 15 characters (e.g., "XX:XX:XX:XX:XX:__")
        prefix = mac[:-2] 
        if prefix not in prefixes:
            prefixes[prefix] = []
        prefixes[prefix].append(mac)
    
    num_aps = len(prefixes)
    print(f"Result: Detected {num_aps} physical AP groups based on MAC prefixes.")
    return num_aps

# --- STRATEGY 2: K-MEANS ON TIME SERIES ---
def strategy_2_kmeans(df, mac_cols):
    print("\n--- Strategy 2: KMeans Clustering by Device ---")
    devices = ['G', 'V', 'R', 'C']
    
    for device in devices:
        device_data = df[df['device_id'] == device][mac_cols]
        if device_data.empty:
            print(f"Skipping Device {device}: No data found.")
            continue
            
        print(f"Analyzing Device: {device} (Timestamps: {len(device_data)})")
        
        # Transpose so rows are MAC addresses and columns are time series tracking
        ts_data = device_data.T 
        
        best_k = 2
        best_score = -1
        k_range = np.arange(20,100) # Testing values around the expected 39
        
        for k in k_range:
            if k >= len(mac_cols): continue
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(ts_data)
            score = silhouette_score(ts_data, labels)
            
            if score > best_score:
                best_score = score
                best_k = k
                
        print(f"  Best K tested for Device {device}: {best_k} (Silhouette Score: {best_score:.4f})")

# --- STRATEGY 3: CORRELATION HISTOGRAM & COMMUNITY DETECTION ---
def strategy_3_correlation_network(df, mac_cols, chosen_device='C', threshold=0.85):
    print(f"\n--- Strategy 3: Mathematical Correlation & Community Detection ({chosen_device}) ---")
    
    device_data = df[df['device_id'] == chosen_device][mac_cols]
    if device_data.empty:
        print(f"Error: No data available for device {chosen_device} to run Strategy 3.")
        return
        
    # Normalize individual signals by their L2 norm energy
    norms = np.linalg.norm(device_data, axis=0)
    norms[norms == 0] = 1 # Avoid dividing by zero
    normalized_data = device_data / norms
    
    # Generate Pearson correlation matrix
    print("  Calculating Correlation Matrix...")
    corr_matrix = normalized_data.corr()
    
    # Extract ONLY the unique pairs (upper triangle, excluding the main diagonal)
    # This prevents the 1.0 self-correlations from warping your histogram metrics
    upper_tri_indices = np.triu_indices_from(corr_matrix, k=1)
    unique_correlations = corr_matrix.values[upper_tri_indices]
    
    # Plotting the requested histogram
    print(f"  Generating correlation histogram inside '{OUTPUT_DIR}/'...")
    plt.figure(figsize=(9, 5))
    plt.hist(unique_correlations, bins=100, color='darkblue', alpha=0.75, edgecolor='black', linewidth=0.2)
    plt.axvline(x=threshold, color='red', linestyle='--', linewidth=1.5, label=f'Current Threshold ({threshold})')
    plt.title(f"Distribution of Unique MAC Correlations (Device {chosen_device})")
    plt.xlabel("Pearson Correlation Coefficient")
    plt.ylabel("Frequency Pair Count")
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    hist_path = os.path.join(OUTPUT_DIR, f"correlation_histogram_{chosen_device}.png")
    plt.savefig(hist_path, dpi=200)
    plt.close()
    print(f"  [SAVED] Histogram visualization written to: {hist_path}")
    
    # Build Network Graph using the designated threshold
    print(f"  Building Graph Network with Threshold >= {threshold}...")
    G = nx.Graph()
    G.add_nodes_from(mac_cols)
    
    for i in range(len(mac_cols)):
        for j in range(i+1, len(mac_cols)):
            mac1, mac2 = mac_cols[i], mac_cols[j]
            corr = corr_matrix.loc[mac1, mac2]
            if corr >= threshold:
                G.add_edge(mac1, mac2, weight=corr)
                
    # Evaluate clusters with Louvain Community Detection
    partition = community_louvain.best_partition(G)
    num_communities = len(set(partition.values()))
    print(f"  Result: At threshold {threshold}, Louvain detected {num_communities} AP clusters.")
    
    # Generate Network layout visualization
    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(G, k=0.18, seed=42)
    cmap = plt.cm.get_cmap('tab20', num_communities)
    nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=25, 
                           cmap=cmap, node_color=list(partition.values()))
    nx.draw_networkx_edges(G, pos, alpha=0.2)
    plt.title(f"Detected AP Communities (Threshold: {threshold} | Found Clusters: {num_communities})")
    plt.axis('off')
    
    net_path = os.path.join(OUTPUT_DIR, f"mac_network_{chosen_device}.png")
    plt.savefig(net_path, dpi=200)
    plt.close()
    print(f"  [SAVED] Network graph written to: {net_path}")

# --- MAIN RUN ---
if __name__ == "__main__":
    # Merge files
    merged_raw = load_and_merge_data()
    
    # Run selection logic and shift parameters
    df_clean, target_macs = preprocess_and_filter(merged_raw)
    
    # Strategy 1
    strategy_1_mac_prefix(target_macs)
    
    # Strategy 2
    strategy_2_kmeans(df_clean, target_macs)
    
    # Strategy 3 (Set your default testing parameters here)
    # You can change 'C' to 'G', 'V', or 'R' and change the threshold parameter here
    strategy_3_correlation_network(df_clean, target_macs, chosen_device='C', threshold=0.4)
    
    print("\n--- Pipeline Execution Finished Successfully ---")