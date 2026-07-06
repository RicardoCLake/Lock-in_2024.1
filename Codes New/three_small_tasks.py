import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.stats
from scipy.special import gamma, hyp2f1
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
OUTPUT_DIR = "./output/tcc_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_FILE = "../Mesures/data03.csv"
TEST_FILES = {
    "I": "../Mesures/data_test_iphone.csv",
    "V2": "../Mesures/data_test_vitor.csv"
}
MAC_RELATION_FILE = "../Mesures/mac_name_relation.csv"

# --- ZONE MAPPING & HELPERS (From original code) ---
ZONE_MAPPING = {
    'LC410_5': 'CN1', 'LC412_5': 'CN1', 'LC413_5': 'CN1',
    'LC414_5': 'CN2', 'LC415_5': 'CN2', 'LC416_5': 'CN2',
    'LC417_5': 'CN3',
    'LC424_5': 'CE1', 'LC426_5': 'CE1',
    'LC437_5': 'CS1',
    'LC442_5': 'CS2',
    'LC443_5': 'CS3', 'LC448_5': 'CS3',
    'LC455_5': 'CO1'
}
for room in ['LC410', 'LC412', 'LC413', 'LC414', 'LC415', 'LC416', 'LC417', 
             'LC424', 'LC426', 'LC437', 'LC442', 'LC443', 'LC448', 'LC455']:
    for part in [1, 2, 3, 4]:
        ZONE_MAPPING[f"{room}_{part}"] = room

def apply_positive_repr(df, mac_cols):
    df_pos = df.copy()
    missing_cols = [col for col in mac_cols if col not in df_pos.columns]
    if missing_cols:
        missing_df = pd.DataFrame(np.nan, index=df_pos.index, columns=missing_cols)
        df_pos = pd.concat([df_pos, missing_df], axis=1)

    df_pos[mac_cols] = df_pos[mac_cols].fillna(-100.0)
    df_pos[mac_cols] = df_pos[mac_cols] + 100.0
    df_pos[mac_cols] = df_pos[mac_cols].clip(lower=0)
    return df_pos

def apply_powed_repr(df_pos, mac_cols):
    df_pow = df_pos.copy()
    df_pow[mac_cols] = (df_pow[mac_cols] / 100.0) ** np.e
    return df_pow

def preprocess_and_map_zones(df):
    df['position'] = df['room'].astype(str) + "_" + df['room_part'].astype(str)
    df['zone'] = df['position'].map(ZONE_MAPPING)
    df = df.dropna(subset=['zone']).reset_index(drop=True)
    return df

# --- TLOC MODEL (From original code) ---
class TLoc:
    def __init__(self, train_data: pd.DataFrame, target_col="zone"):
        self.train_data = train_data
        self.target_col = target_col
        self.aps = [col for col in train_data.columns if col != target_col]
        
        if len(self.aps) == 1:
            self.max_power = int(self.train_data[self.aps].max())
        else:
            self.max_power = int(self.train_data[self.aps].max().max())

        self.spaces = list(self.train_data[self.target_col].unique())
        self.power_probability_masks = {}
        self.power_prior_probability_distribution = {}
        self.eps = 1e-5

    def get_mu_and_phi_estimation(self, data, router):
        mu, phi = [], []
        data_of_router = data[[self.target_col, router]]
        for space in self.spaces:
            data_of_router_in_space = data_of_router[data_of_router[self.target_col] == space]
            data_of_router_in_space_no_zero = data_of_router_in_space[data_of_router_in_space[router] != 0]
            
            if len(data_of_router_in_space_no_zero) == 0:
                mu.append(0.0)
            else:
                mu.append(data_of_router_in_space_no_zero[router].mean())
            phi.append(1 - data_of_router_in_space_no_zero.shape[0] / max(1, data_of_router_in_space.shape[0]))
        return mu, phi

    def train(self):
        for router in self.aps:
            self.power_probability_masks[router] = {}
            self.power_prior_probability_distribution[router] = {}
            mu, phi = self.get_mu_and_phi_estimation(self.train_data, router)
            total_num_samples_in_router = self.train_data[router].shape[0]
            
            for power in range(0, self.max_power + 1):
                self.power_probability_masks[router][power] = self.approximate_position_density_function_given_router(
                    power, np.array(mu), np.array(phi)
                )
                num_samples_with_value_power_in_router = (self.train_data[router] == power).sum()
                self.power_prior_probability_distribution[router][power] = num_samples_with_value_power_in_router / max(1, total_num_samples_in_router)

    def cumulative_distribution_function_of_t_student(self, x, v):
        return 0.5 + x * gamma((v + 1) / 2) * hyp2f1(1 / 2, (v + 1) / 2, 3 / 2, -(x ** 2) / v) / (
                np.sqrt(v * np.pi) * gamma(v / 2))

    def cumulative_distribution_function_of_power(self, power, mu, phi, sigma, v):
        return phi * np.heaviside(power, 1) + (1 - phi) * self.cumulative_distribution_function_of_t_student((power - mu) / sigma, v)

    def approximate_position_density_function_given_router(self, power, mu, phi, sigma=5, num_samples_per_ap=30, t_score_alpha=0.05):
        v = np.ceil(num_samples_per_ap * (1 - phi) - 1)
        v = np.where(v <= 0, 1, v)
        t_score = scipy.stats.t.ppf(0.5 + t_score_alpha, v)
        density_function = self.cumulative_distribution_function_of_power(
            power + t_score * sigma, mu, phi, sigma, v) - self.cumulative_distribution_function_of_power(
            power - t_score * sigma, mu, phi, sigma, v)
        return density_function

    def pred(self, X_test):
        y_pred = []
        all_probs = [] # To store the probabilities
        min_prob = self.eps * np.ones(len(self.spaces))

        for _, test_sample in X_test.iterrows():
            distribution_xy_given_bf = np.ones(len(self.spaces))
            for router in self.aps:
                power = int(test_sample[router])
                try:
                    prob_p_given_xybfr = self.power_probability_masks[router][power]
                except KeyError:
                    prob_p_given_xybfr = min_prob
                prob_p_given_xybfr = np.maximum(prob_p_given_xybfr, min_prob)
                distribution_xy_given_bf = distribution_xy_given_bf * prob_p_given_xybfr

            # Normalize the distribution so it sums to 1 (proper probabilities)
            prob_sum = distribution_xy_given_bf.sum()
            if prob_sum > 0:
                normalized_probs = distribution_xy_given_bf / prob_sum
            else:
                normalized_probs = distribution_xy_given_bf
                
            all_probs.append(normalized_probs)
            
            room_pred = self.spaces[distribution_xy_given_bf.argmax()]
            y_pred.append(room_pred)

        y_pred = np.array(y_pred)
        all_probs = np.array(all_probs)
        return all_probs, y_pred, None # Returning probabilities as the first element

# --- TASK 1: SINGLE CAMPAIGN DATA SPLIT ---
def task1_single_campaign_split(df_raw, mac_cols):
    print("\n--- Task 1: Evaluating the necessity of the second campaign ---")
    df = preprocess_and_map_zones(df_raw)
    df_pos = apply_positive_repr(df, mac_cols)
    
    # We will stratify by the 'position' (room + room_part)
    X = df_pos[mac_cols]
    y_target = df_pos['zone']
    y_stratify = df_pos['position']
    
    train_percentages = np.arange(0.05, 0.96, 0.05)
    accuracies = []
    f1_scores = []
    
    knn = KNeighborsClassifier(n_neighbors=5, weights='distance', metric='euclidean')

    for train_size in train_percentages:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_target, train_size=train_size, stratify=y_stratify, random_state=42
        )
        knn.fit(X_train, y_train)
        preds = knn.predict(X_test)
        
        # Weighted Accuracy (approximated by Balanced Accuracy) and Macro F1
        acc = balanced_accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average='macro', zero_division=0)
        
        accuracies.append(acc)
        f1_scores.append(f1)
        print(f"Split Train {int(train_size*100)}% -> Bal Acc: {acc:.3f} | F1: {f1:.3f}")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(train_percentages * 100, accuracies, label='Balanced Accuracy', marker='o', linewidth=2, color='#FF6F61')
    plt.plot(train_percentages * 100, f1_scores, label='Macro F1-Score', marker='s', linewidth=2, color='#32CD32')
    
    plt.title("Metrics vs. Training Split Percentage (Single Campaign Validation)", fontsize=16, fontweight='bold')
    plt.xlabel("Training Data Split Percentage (%)", fontsize=14)
    plt.ylabel("Score", fontsize=14)
    plt.xticks(np.arange(5, 100, 5))
    plt.ylim(0.8, 1.05) # Because single dataset usually yields artificially high scores
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=14)
    plt.tight_layout()
    
    plot_path = os.path.join(OUTPUT_DIR, "task1_single_campaign_split.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[SAVED] Task 1 plot exported to {plot_path}")


# --- TASK 2: DOORS EFFECT ---
def task2_doors_effect(df_train_raw, test_datasets_raw, mac_cols):
    print("\n--- Task 2: Doors Effect ---")
    
    def prepare_doors_data(df):
        # Remove room_part == 5 (corridors) as we only care about inside the room
        df_filtered = df[df['room_part'] != 5].copy()
        # Ensure we have a valid door_status target
        df_filtered = df_filtered.dropna(subset=['door_status']).reset_index(drop=True)
        return df_filtered

    df_train = prepare_doors_data(df_train_raw)
    
    train_pos = apply_positive_repr(df_train, mac_cols)
    train_pow = apply_powed_repr(train_pos, mac_cols)
    
    X_train = train_pow[mac_cols]
    y_train = df_train['door_status']

    # WKNN Pow Sorensen (braycurtis)
    knn = KNeighborsClassifier(n_neighbors=5, weights='distance', metric='braycurtis')
    knn.fit(X_train, y_train)

    md_content = "### Task 2: Doors Effect Results\n\n"
    md_content += "Evaluation of Door Status inside rooms using WKNN (Powered, Sorensen/Bray-Curtis).\n\n"
    md_content += "| Test Device | Balanced Accuracy | Macro F1-Score |\n"
    md_content += "| :--- | :--- | :--- |\n"

    for dev, df_test_raw in test_datasets_raw.items():
        df_test = prepare_doors_data(df_test_raw)
        
        test_pos = apply_positive_repr(df_test, mac_cols)
        test_pow = apply_powed_repr(test_pos, mac_cols)
        
        X_test = test_pow[mac_cols]
        y_test = df_test['door_status']
        
        preds = knn.predict(X_test)
        
        acc = balanced_accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average='macro', zero_division=0)
        
        print(f"Device {dev} -> Bal Acc: {acc:.3f} | F1: {f1:.3f}")
        md_content += f"| **{dev}** | {acc:.4f} | {f1:.4f} |\n"

    md_path = os.path.join(OUTPUT_DIR, "task2_doors_effect.md")
    with open(md_path, "w") as f:
        f.write(md_content)
    print(f"[SAVED] Task 2 table exported to {md_path}")


# --- TASK 3: TEMPORAL AVERAGE ---
def task3_temporal_average(df_train_raw, test_datasets_raw, mac_cols):
    print("\n--- Task 3: Temporal Average ---")
    
    df_train = preprocess_and_map_zones(df_train_raw)
    train_pos = apply_positive_repr(df_train, mac_cols)
    
    print("Training TLoc with 409 MAC addresses...")
    tloc = TLoc(train_pos[mac_cols].join(df_train[['zone']]), target_col='zone')
    tloc.train()

    windows = list(range(1, 21))
    results = {dev: [] for dev in test_datasets_raw.keys()}

    for dev, df_test_raw in test_datasets_raw.items():
        print(f"Evaluating temporal sequence for Device {dev}...")
        df_test = preprocess_and_map_zones(df_test_raw)
        test_pos = apply_positive_repr(df_test, mac_cols)
        
        # 1. Get raw predictions for all samples
        _, preds, _ = tloc.pred(test_pos[mac_cols].join(df_test[['zone']]))
        df_test['pred'] = preds
        
        # 2. Iterate over window sizes
        for w in windows:
            y_true_seq = []
            y_pred_seq = []
            
            # Group by actual class (zone) to respect spatial bounds
            for zone in df_test['zone'].unique():
                zone_data = df_test[df_test['zone'] == zone]
                
                # Take sequential chunks of size w
                for i in range(0, len(zone_data), w):
                    chunk = zone_data.iloc[i:i+w]
                    if len(chunk) > 0:
                        # Voting mechanism: most frequent prediction in chunk
                        majority_pred = chunk['pred'].mode()[0]
                        y_true_seq.append(zone)
                        y_pred_seq.append(majority_pred)
            
            f1 = f1_score(y_true_seq, y_pred_seq, average='macro', zero_division=0)
            results[dev].append(f1)

    # Plot
    plt.figure(figsize=(10, 6))
    
    colors = {'I': '#1f77b4', 'V2': '#ff7f0e'}
    markers = {'I': 'o', 'V2': 's'}
    
    for dev, f1_list in results.items():
        plt.plot(windows, f1_list, label=f'Device {dev}', marker=markers[dev], linewidth=2, color=colors[dev])
        
    plt.title("Macro F1-Score vs. Temporal Window Size (TLoc Model)", fontsize=16, fontweight='bold')
    plt.xlabel("Window Size (Number of Sequential Samples)", fontsize=14)
    plt.ylabel("Macro F1-Score", fontsize=14)
    plt.xticks(windows)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=14)
    plt.tight_layout()
    
    plot_path = os.path.join(OUTPUT_DIR, "task3_temporal_average.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[SAVED] Task 3 plot exported to {plot_path}")

# --- TASK 4: TEMPORAL AVERAGE (PROBABILITY SOFT VOTING) ---
def task4_temporal_average_prob(df_train_raw, test_datasets_raw, mac_cols):
    print("\n--- Task 4: Temporal Average (Probability Soft Voting) ---")
    
    df_train = preprocess_and_map_zones(df_train_raw)
    train_pos = apply_positive_repr(df_train, mac_cols)
    
    print("Training TLoc with 409 MAC addresses...")
    tloc = TLoc(train_pos[mac_cols].join(df_train[['zone']]), target_col='zone')
    tloc.train()

    windows = list(range(1, 21))
    results = {dev: [] for dev in test_datasets_raw.keys()}

    for dev, df_test_raw in test_datasets_raw.items():
        print(f"Evaluating temporal probability sequence for Device {dev}...")
        df_test = preprocess_and_map_zones(df_test_raw)
        test_pos = apply_positive_repr(df_test, mac_cols)
        
        # 1. Get probability distributions for all samples
        y_probs, _, _ = tloc.pred(test_pos[mac_cols].join(df_test[['zone']]))
        
        # 2. Iterate over window sizes
        for w in windows:
            y_true_seq = []
            y_pred_seq = []
            
            # Group by actual class (zone) to respect spatial bounds
            for zone in df_test['zone'].unique():
                # Get the indices of the samples belonging to this zone
                zone_indices = df_test[df_test['zone'] == zone].index
                
                # Take sequential chunks of size w
                for i in range(0, len(zone_indices), w):
                    chunk_idx = zone_indices[i:i+w]
                    if len(chunk_idx) > 0:
                        ## Soft voting mechanism: average the probability vectors
                        chunk_probs = y_probs[chunk_idx]
                        
                        # Arithmetic mean (current)
                        avg_probs = np.mean(chunk_probs, axis=0)

                        # Harmonic mean
                        harmonic_probs = len(chunk_probs) / np.sum(1.0 / chunk_probs, axis=0)

                        # Geometric mean
                        geometric_probs = np.exp(np.mean(np.log(chunk_probs), axis=0))

                        # Quadratic mean (Root Mean Square, RMS)
                        quadratic_probs = np.sqrt(np.mean(chunk_probs**2, axis=0))
                        
                        # The prediction is the class with the highest average probability
                        best_class_idx = np.argmax(quadratic_probs)
                        majority_pred = tloc.spaces[best_class_idx]
                        
                        y_true_seq.append(zone)
                        y_pred_seq.append(majority_pred)
            
            f1 = f1_score(y_true_seq, y_pred_seq, average='macro', zero_division=0)
            results[dev].append(f1)

    # Plot
    plt.figure(figsize=(10, 6))
    
    colors = {'I': '#1f77b4', 'V2': '#ff7f0e'}
    markers = {'I': 'o', 'V2': 's'}
    
    for dev, f1_list in results.items():
        plt.plot(windows, f1_list, label=f'Device {dev} (Soft Voting)', marker=markers[dev], linewidth=2, color=colors[dev], linestyle='--')
        
    plt.title("Macro F1-Score vs. Temporal Window Size (Soft Voting - Probabilities)", fontsize=14, fontweight='bold')
    plt.xlabel("Window Size (Number of Sequential Samples)", fontsize=12)
    plt.ylabel("Macro F1-Score", fontsize=12)
    plt.xticks(windows)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    plot_path = os.path.join(OUTPUT_DIR, "task4_temporal_average_prob.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[SAVED] Task 4 plot exported to {plot_path}")

# --- MAIN PIPELINE ---
if __name__ == "__main__":
    print("Loading Data...")
    df_train_raw = pd.read_csv(TRAIN_FILE, index_col=0)
    
    test_datasets_raw = {}
    for dev, file_path in TEST_FILES.items():
        test_datasets_raw[dev] = pd.read_csv(file_path, index_col=0)
        
    relation = pd.read_csv(MAC_RELATION_FILE, index_col=0)
    meta_cols = ['timestamp', 'room', 'device_id', 'door_status', 'room_part']
    all_macs = [c for c in df_train_raw.columns if c not in meta_cols]
    
    target_ssids = ["Guest-CentraleSupelec", "eduroam", 'stop&go', 'CD91', 'fabrique2024']
    good_aps = relation[relation['ap_name'].isin(target_ssids)]["ap_mac"].to_list()
    filtered_macs = [mac for mac in good_aps if mac in all_macs]
    
    print(f"Total MACs extracted for filtered subset: {len(filtered_macs)}")

    # Execute Tasks
    task1_single_campaign_split(df_train_raw, filtered_macs)
    #task2_doors_effect(df_train_raw, test_datasets_raw, filtered_macs)
    task3_temporal_average(df_train_raw, test_datasets_raw, filtered_macs)
    task4_temporal_average_prob(df_train_raw, test_datasets_raw, filtered_macs) # <--- NEW TASK ADDED HERE

    print("\n--- All requested analyses complete! Boa sorte com o TCC! ---")