import pandas as pd
import numpy as np
import os
import gc
import scipy.stats
from scipy.special import gamma, hyp2f1
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (balanced_accuracy_score, recall_score, precision_score, f1_score)
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.spatial.distance import cdist
import warnings

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
OUTPUT_DIR = "./output/ml"
os.makedirs(OUTPUT_DIR, exist_ok=True)
RESULTS_MD = os.path.join(OUTPUT_DIR, "augmented_results.md")

TRAIN_FILE = "../Mesures/data03.csv"
TEST_FILES = {
    "I": "../Mesures/data_test_iphone.csv",
    "V2": "../Mesures/data_test_vitor.csv"
}
MAC_RELATION_FILE = "../Mesures/mac_name_relation.csv"

# --- 1. OPTIMIZED CUSTOM MODELS ---

class FastSorensenKNN(BaseEstimator, ClassifierMixin):
    """
    A mathematically equivalent but heavily vectorized KNN for Sorensen (Bray-Curtis) 
    distances, specifically optimized for strictly positive data arrays.
    """
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        # np.asarray safely catches Pandas Series, DataFrames, and StringArrays
        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y)
        self.X_train_sum = self.X_train.sum(axis=1) 
        return self

    def predict(self, X):
        X_test = np.asarray(X)
        y_pred = []
        
        chunk_size = 100 
        for i in range(0, X_test.shape[0], chunk_size):
            X_chunk = X_test[i:i+chunk_size]
            
            l1_dist = cdist(X_chunk, self.X_train, metric='cityblock')
            
            chunk_sum = X_chunk.sum(axis=1)[:, np.newaxis] 
            denominator = chunk_sum + self.X_train_sum     
            denominator[denominator == 0] = 1e-10 
            
            bray_curtis_dist = l1_dist / denominator
            
            knn_indices = np.argpartition(bray_curtis_dist, self.n_neighbors, axis=1)[:, :self.n_neighbors]
            knn_distances = np.take_along_axis(bray_curtis_dist, knn_indices, axis=1)
            
            weights = 1.0 / (knn_distances + 1e-10)
            knn_labels = self.y_train[knn_indices]
            
            for j in range(X_chunk.shape[0]):
                unique_labels = np.unique(knn_labels[j])
                weighted_votes = np.zeros(len(unique_labels))
                for k, lbl in enumerate(unique_labels):
                    weighted_votes[k] = np.sum(weights[j][knn_labels[j] == lbl])
                y_pred.append(unique_labels[np.argmax(weighted_votes)])
                
        return np.array(y_pred)

class TLoc:
    def __init__(self, train_data: pd.DataFrame, target_col="zone"):
        self.train_data = train_data
        self.target_col = target_col
        self.aps = [col for col in train_data.columns if col != target_col]
        self.max_power = int(self.train_data[self.aps].max().max()) if len(self.aps) > 1 else int(self.train_data[self.aps].max())
        self.spaces = list(self.train_data[self.target_col].unique())
        self.power_probability_masks = {}
        self.eps = 1e-5

    def cumulative_distribution_function_of_t_student(self, x, v):
        return 0.5 + x * gamma((v + 1) / 2) * hyp2f1(1 / 2, (v + 1) / 2, 3 / 2, -(x ** 2) / v) / (np.sqrt(v * np.pi) * gamma(v / 2))

    def train(self):
        print("    > [TLoc] Pre-computing NumPy matrices for blazing fast training...")
        X = self.train_data[self.aps].values
        y = self.train_data[self.target_col].values
        
        unique_zones, y_idx = np.unique(y, return_inverse=True)
        num_zones = len(unique_zones)
        space_counts = np.maximum(1, np.bincount(y_idx, minlength=num_zones))
        
        zone_to_idx = {z: j for j, z in enumerate(unique_zones)}
        target_indices = [zone_to_idx[s] for s in self.spaces]

        powers = np.arange(self.max_power + 1).reshape(-1, 1) 
        num_samples_per_ap = 30
        t_score_alpha = 0.05
        sigma = 5

        for i, router in enumerate(self.aps):
            self.power_probability_masks[router] = {}
            router_vals = X[:, i]
            mask_nz = router_vals != 0
            
            nz_counts = np.bincount(y_idx[mask_nz], minlength=num_zones)
            nz_sums = np.bincount(y_idx[mask_nz], weights=router_vals[mask_nz], minlength=num_zones)
            
            nz_means = np.zeros(num_zones)
            valid = nz_counts > 0
            nz_means[valid] = nz_sums[valid] / nz_counts[valid]
            
            phi = 1.0 - (nz_counts / space_counts)
            
            phi_arr = phi[target_indices]
            mu_arr = nz_means[target_indices]
            
            v = np.where(np.ceil(num_samples_per_ap * (1 - phi_arr) - 1) <= 0, 1, 
                         np.ceil(num_samples_per_ap * (1 - phi_arr) - 1))
            t_score = scipy.stats.t.ppf(0.5 + t_score_alpha, v)
            
            upper_bound = powers + t_score * sigma
            lower_bound = powers - t_score * sigma
            
            cdf_upper = phi_arr * np.heaviside(upper_bound, 1) + (1 - phi_arr) * self.cumulative_distribution_function_of_t_student((upper_bound - mu_arr) / sigma, v)
            cdf_lower = phi_arr * np.heaviside(lower_bound, 1) + (1 - phi_arr) * self.cumulative_distribution_function_of_t_student((lower_bound - mu_arr) / sigma, v)
            
            density_matrix = cdf_upper - cdf_lower
            for p in range(self.max_power + 1):
                self.power_probability_masks[router][p] = density_matrix[p]

    def pred(self, X_test):
        y_pred = []
        min_prob = self.eps * np.ones(len(self.spaces))
        X_mat = X_test[self.aps].values 
        
        for i in range(X_mat.shape[0]):
            distribution = np.ones(len(self.spaces))
            for j, router in enumerate(self.aps):
                power = int(X_mat[i, j])
                prob = self.power_probability_masks[router].get(power, min_prob)
                distribution *= np.maximum(prob, min_prob)
            y_pred.append(self.spaces[distribution.argmax()])
            
        y_pred = np.array(y_pred)
        if self.target_col in X_test.columns:
            ground_truth = np.array(list(X_test[self.target_col]))
            return np.sum(y_pred == ground_truth) / len(ground_truth), y_pred, ground_truth
        return None, y_pred, None

# --- 2. PREPROCESSING & DATA AUGMENTATION ---
ZONE_MAPPING = {
    'LC410_5': 'CN1', 'LC412_5': 'CN1', 'LC413_5': 'CN1',
    'LC414_5': 'CN2', 'LC415_5': 'CN2', 'LC416_5': 'CN2',
    'LC417_5': 'CN3', 'LC424_5': 'CE1', 'LC426_5': 'CE1',
    'LC437_5': 'CS1', 'LC442_5': 'CS2', 'LC443_5': 'CS3', 'LC448_5': 'CS3',
    'LC455_5': 'CO1'
}
for room in ['LC410', 'LC412', 'LC413', 'LC414', 'LC415', 'LC416', 'LC417', 
             'LC424', 'LC426', 'LC437', 'LC442', 'LC443', 'LC448', 'LC455']:
    for part in [1, 2, 3, 4]:
        ZONE_MAPPING[f"{room}_{part}"] = room

def preprocess_and_map_zones(df):
    df['position'] = df['room'].astype(str) + "_" + df['room_part'].astype(str)
    df['zone'] = df['position'].map(ZONE_MAPPING)
    return df.dropna(subset=['zone']).reset_index(drop=True)

def apply_positive_repr(df, mac_cols):
    df_pos = df.copy()
    missing_cols = [col for col in mac_cols if col not in df_pos.columns]
    if missing_cols:
        df_pos = pd.concat([df_pos, pd.DataFrame(np.nan, index=df_pos.index, columns=missing_cols)], axis=1)
    df_pos[mac_cols] = df_pos[mac_cols].fillna(-100.0) + 100.0
    df_pos[mac_cols] = df_pos[mac_cols].clip(lower=0).astype(np.float32)
    return df_pos

def apply_powed_repr(df_pos, mac_cols):
    df_pow = df_pos.copy()
    df_pow[mac_cols] = ((df_pow[mac_cols] / 100.0) ** np.e).astype(np.float32)
    return df_pow

def augment_data(df, mac_cols):
    prefixes = {}
    for mac in mac_cols:
        prefix = mac[:-2] 
        if prefix not in prefixes:
            prefixes[prefix] = []
        prefixes[prefix].append(mac)
    
    augmented_dfs = [df]
    print(f"  > Detected {len(prefixes)} distinct APs based on mac[:-2]. Augmenting data {len(prefixes) + 1}x...")
    
    for prefix, macs_to_drop in prefixes.items():
        df_broken = df.copy()
        df_broken[macs_to_drop] = np.nan
        augmented_dfs.append(df_broken)
        
    aug_df = pd.concat(augmented_dfs, ignore_index=True)
    del augmented_dfs
    gc.collect() 
    return aug_df

def log_results(model_name, test_dev, y_true, y_pred, results_list):
    results_list.append({
        "Test Device": test_dev,
        "Model": model_name,
        "Balanced Accuracy": f"{balanced_accuracy_score(y_true, y_pred):.4f}",
        "Macro Precision": f"{precision_score(y_true, y_pred, average='macro', zero_division=0):.4f}",
        "Macro Recall": f"{recall_score(y_true, y_pred, average='macro', zero_division=0):.4f}",
        "Macro F1": f"{f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}"
    })

# --- 3. MAIN PIPELINE ---
if __name__ == "__main__":
    print("--- Loading and Filtering Data ---")
    df_train_raw = pd.read_csv(TRAIN_FILE, index_col=0)
    
    relation = pd.read_csv(MAC_RELATION_FILE, index_col=0)
    target_ssids = ["Guest-CentraleSupelec", "eduroam", 'stop&go', 'CD91', 'fabrique2024']
    good_aps = relation[relation['ap_name'].isin(target_ssids)]["ap_mac"].to_list()
    
    # 1. Identify valid MACs
    meta_cols = ['timestamp', 'room', 'device_id', 'door_status', 'room_part']
    all_macs = [c for c in df_train_raw.columns if c not in meta_cols]
    filtered_macs = [mac for mac in good_aps if mac in all_macs]
    
    print(f"  > Selected {len(filtered_macs)} target MAC addresses.")
    
    # --- FIX: Aggressively drop unused MAC columns to save RAM and correct shapes ---
    keep_cols = [c for c in meta_cols if c in df_train_raw.columns] + filtered_macs
    df_train_raw = df_train_raw[keep_cols]
    
    df_train = preprocess_and_map_zones(df_train_raw)
    del df_train_raw
    gc.collect()
    
    print("--- Augmenting Training Data ---")
    df_train_aug = augment_data(df_train, filtered_macs)
    print(f"  > Original Shape: {df_train.shape} | Augmented Shape: {df_train_aug.shape}")
    del df_train
    gc.collect()

    print("--- Calculating Representations ---")
    train_pos = apply_positive_repr(df_train_aug, filtered_macs)
    train_pow = apply_powed_repr(train_pos, filtered_macs)
    
    le = LabelEncoder()
    y_train = df_train_aug['zone'].values
    y_train_encoded = le.fit_transform(y_train)

    zone_col_backup = df_train_aug[['zone']].copy()
    del df_train_aug
    gc.collect()

    tests = {}
    for test_dev, file_path in TEST_FILES.items():
        df_test_raw = pd.read_csv(file_path, index_col=0)
        # Drop unwanted MACs in test data as well
        keep_cols_test = [c for c in meta_cols if c in df_test_raw.columns] + [c for c in filtered_macs if c in df_test_raw.columns]
        df_test_raw = df_test_raw[keep_cols_test]
        
        df_test = preprocess_and_map_zones(df_test_raw)
        tests[test_dev] = {
            "y": df_test['zone'].values,
            "raw_zone": df_test[['zone']],
            "pos": apply_positive_repr(df_test, filtered_macs),
            "pow": apply_powed_repr(apply_positive_repr(df_test, filtered_macs), filtered_macs)
        }

    results = []

    # --- 4. SEQUENTIAL TRAINING & EVALUATION ---
    print("\n--- Training Models (Sequentially to avoid OOM) ---")

    # 1. TLoc
    print("  > Training TLoc...")
    model = TLoc(train_pos[filtered_macs].join(zone_col_backup), target_col='zone')
    model.train()
    for test_dev, test_data in tests.items():
        _, preds, _ = model.pred(test_data["pos"][filtered_macs].join(test_data["raw_zone"]))
        log_results("TLoc", test_dev, test_data["y"], preds, results)
    del model; gc.collect() 

    # 2. XGBoost
    print("  > Training XGBoost...")
    model = XGBClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
    model.fit(train_pos[filtered_macs], y_train_encoded)
    for test_dev, test_data in tests.items():
        preds = le.inverse_transform(model.predict(test_data["pos"][filtered_macs]))
        log_results("XGBoost", test_dev, test_data["y"], preds, results)
    del model; gc.collect()

    # 3-6. WKNN Models
    knn_configs = [
        ("WKNN_Pos_Euclid", train_pos, "pos", "euclidean"),
        ("WKNN_Pow_Euclid", train_pow, "pow", "euclidean"),
        ("WKNN_Pos_Sorensen", train_pos, "pos", "sorensen_fast"),
        ("WKNN_Pow_Sorensen", train_pow, "pow", "sorensen_fast")
    ]

    for model_name, train_df, test_repr_key, metric in knn_configs:
        print(f"  > Training {model_name}...")
        
        if metric == "sorensen_fast":
            model = FastSorensenKNN(n_neighbors=5)
        else:
            model = KNeighborsClassifier(n_neighbors=5, weights='distance', metric=metric, n_jobs=-1)
            
        model.fit(train_df[filtered_macs], y_train)
        
        for test_dev, test_data in tests.items():
            preds = model.predict(test_data[test_repr_key][filtered_macs])
            log_results(model_name, test_dev, test_data["y"], preds, results)
            
        del model; gc.collect()

    # --- 5. EXPORT MARKDOWN TABLE ---
    res_df = pd.DataFrame(results)
    
    md_table = "| Test Device | Model | Balanced Accuracy | Macro Precision | Macro Recall | Macro F1 |\n"
    md_table += "| :--- | :--- | :--- | :--- | :--- | :--- |\n"
    for _, row in res_df.iterrows():
        md_table += f"| {row['Test Device']} | {row['Model']} | {row['Balanced Accuracy']} | {row['Macro Precision']} | {row['Macro Recall']} | {row['Macro F1']} |\n"

    with open(RESULTS_MD, "w") as f:
        f.write("# Augmented Training Results (409 MACs)\n\n")
        f.write("*Training Dataset augmented to simulate AP failures.* \n\n")
        f.write(md_table)
    
    print(f"\n[SAVED] Results successfully exported to {RESULTS_MD}")