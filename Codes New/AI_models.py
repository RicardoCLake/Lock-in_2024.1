import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.stats
from scipy.special import gamma, hyp2f1
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (balanced_accuracy_score, recall_score, precision_score, 
                             f1_score, confusion_matrix, ConfusionMatrixDisplay)
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
OUTPUT_DIR = "./output/ml"
os.makedirs(OUTPUT_DIR, exist_ok=True)
RESULTS_CSV = os.path.join(OUTPUT_DIR, "classification_results.csv")
CM_PREDS_CSV = os.path.join(OUTPUT_DIR, "tloc_v2_preds.csv")

TRAIN_FILE = "../Mesures/data03.csv"
TEST_FILES = {
    "I": "../Mesures/data_test_iphone.csv",
    "V2": "../Mesures/data_test_vitor.csv"
}
MAC_RELATION_FILE = "../Mesures/mac_name_relation.csv"

# --- 1. TLOC MODEL DEFINITION ---
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
        mu = []
        phi = []
        data_of_router = data[[self.target_col, router]]
        for space in self.spaces:
            data_of_router_in_space = data_of_router[data_of_router[self.target_col] == space]
            data_of_router_in_space_without_zero_values = data_of_router_in_space[data_of_router_in_space[router] != 0]
            
            if len(data_of_router_in_space_without_zero_values) == 0:
                mu.append(0.0)
            else:
                mu.append(data_of_router_in_space_without_zero_values[router].mean())
                
            phi.append(1 - data_of_router_in_space_without_zero_values.shape[0] / max(1, data_of_router_in_space.shape[0]))

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
        cdf = phi * np.heaviside(power, 1) + (1 - phi) * self.cumulative_distribution_function_of_t_student(
            (power - mu) / sigma, v)
        return cdf

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

            room_pred = self.spaces[distribution_xy_given_bf.argmax()]
            y_pred.append(room_pred)

        y_pred = np.array(y_pred)
        if self.target_col in X_test.columns:
            ground_truth = np.array(list(X_test[self.target_col]))
            ac = np.sum(y_pred == ground_truth) / len(ground_truth)
            return ac, y_pred, ground_truth
        return None, y_pred, None

# --- 2. ZONE MAPPING DEFINITION ---
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

# --- 3. DATA REPRESENTATION FUNCTIONS ---
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

# --- 4. PIPELINE EXECUTION ---
def run_pipeline():
    print("--- Loading Datasets ---")
    df_train_raw = pd.read_csv(TRAIN_FILE, index_col=0)
    
    meta_cols = ['timestamp', 'room', 'device_id', 'door_status', 'room_part']
    all_macs = [c for c in df_train_raw.columns if c not in meta_cols]
    
    relation = pd.read_csv(MAC_RELATION_FILE, index_col=0)
    target_ssids = ["Guest-CentraleSupelec", "eduroam", 'stop&go', 'CD91', 'fabrique2024']
    good_aps = relation[relation['ap_name'].isin(target_ssids)]["ap_mac"].to_list()
    filtered_macs = [mac for mac in good_aps if mac in all_macs]
    
    df_train = preprocess_and_map_zones(df_train_raw)
    test_datasets = {}
    for dev, file_path in TEST_FILES.items():
        df_test_raw = pd.read_csv(file_path, index_col=0)
        test_datasets[dev] = preprocess_and_map_zones(df_test_raw)
        
    le = LabelEncoder()
    le.fit(df_train['zone'])

    results = []
    mac_strategies = {
        "All_MACs": all_macs,
        "Filtered_MACs": filtered_macs
    }

    for strategy_name, mac_cols in mac_strategies.items():
        print(f"\nEvaluating Strategy: {strategy_name} ({len(mac_cols)} features)")
        
        train_pos = apply_positive_repr(df_train, mac_cols)
        train_pow = apply_powed_repr(train_pos, mac_cols)
        
        y_train = df_train['zone'].values
        y_train_encoded = le.transform(y_train)

        models = {
            "TLoc": TLoc(train_pos[mac_cols].join(df_train[['zone']]), target_col='zone'), 
            "XGBoost": XGBClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1),
            "WKNN_Pos_Euclid": KNeighborsClassifier(n_neighbors=5, weights='distance', metric='euclidean'),
            "WKNN_Pow_Euclid": KNeighborsClassifier(n_neighbors=5, weights='distance', metric='euclidean'),
            "WKNN_Pos_Sorensen": KNeighborsClassifier(n_neighbors=5, weights='distance', metric='braycurtis'),
            "WKNN_Pow_Sorensen": KNeighborsClassifier(n_neighbors=5, weights='distance', metric='braycurtis')
        }

        print("  Training models...")
        models["TLoc"].train()
        models["XGBoost"].fit(train_pos[mac_cols], y_train_encoded)
        models["WKNN_Pos_Euclid"].fit(train_pos[mac_cols], y_train)
        models["WKNN_Pow_Euclid"].fit(train_pow[mac_cols], y_train)
        models["WKNN_Pos_Sorensen"].fit(train_pos[mac_cols], y_train)
        models["WKNN_Pow_Sorensen"].fit(train_pow[mac_cols], y_train)

        for test_dev, df_test in test_datasets.items():
            print(f"  Testing on Device: {test_dev}")
            
            test_pos = apply_positive_repr(df_test, mac_cols)
            test_pow = apply_powed_repr(test_pos, mac_cols)
            y_test = df_test['zone'].values
            
            for model_name, model in models.items():
                if "Pow" in model_name:
                    X_test = test_pow[mac_cols]
                else:
                    X_test = test_pos[mac_cols]

                if model_name == "TLoc":
                    _, preds, _ = model.pred(test_pos[mac_cols].join(df_test[['zone']])) 
                elif model_name == "XGBoost":
                    preds_enc = model.predict(X_test)
                    preds = le.inverse_transform(preds_enc)
                else:
                    preds = model.predict(X_test)
                
                if model_name == "TLoc" and test_dev == "V2" and strategy_name == "Filtered_MACs":
                    cm_df = pd.DataFrame({"y_true": y_test, "y_pred": preds})
                    cm_df.to_csv(CM_PREDS_CSV, index=False)
                
                bal_acc = balanced_accuracy_score(y_test, preds)
                rec = recall_score(y_test, preds, average='macro', zero_division=0)
                prec = precision_score(y_test, preds, average='macro', zero_division=0)
                f1 = f1_score(y_test, preds, average='macro', zero_division=0)
                
                results.append({
                    "Strategy": strategy_name,
                    "Test_Device": test_dev,
                    "Model": model_name,
                    "Balanced_Accuracy": bal_acc,
                    "Macro_Recall": rec,
                    "Macro_Precision": prec,
                    "Macro_F1": f1
                })

    res_df = pd.DataFrame(results)
    res_df.to_csv(RESULTS_CSV, index=False)
    print(f"\n[SAVED] Results exported to {RESULTS_CSV}")
    return res_df

# --- 5. PLOTTING ---
def generate_plots():
    if not os.path.exists(RESULTS_CSV):
        res_df = run_pipeline()
    else:
        print(f"Loading cached results from {RESULTS_CSV}")
        res_df = pd.read_csv(RESULTS_CSV)

    # 1. Precision x Recall Plot
    # Adjusted figsize to be slightly wider to accommodate square aspect subplots gracefully
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    devices = ['V2', 'I']
    
    models = res_df['Model'].unique()
    
    # Vibrant "happy" colors for the unique models
    happy_colors = ['#FF6F61', '#32CD32', '#00F5FF', '#FFD700', '#DA70D6', '#FF69B4'] 
    color_map = dict(zip(models, happy_colors[:len(models)]))
    marker_map = {"All_MACs": "o", "Filtered_MACs": "*"}

    # --- NEW: Compute uniform global boundaries across both axes and metrics ---
    all_values = pd.concat([res_df['Macro_Recall'], res_df['Macro_Precision']])
    axis_min = max(0.0, all_values.min() - 0.05)  # Pad slightly below but don't drop under 0
    axis_max = min(1.0, all_values.max() + 0.05)  # Pad slightly above but don't exceed 1

    for idx, dev in enumerate(devices):
        ax = axes[idx]
        subset_dev = res_df[res_df['Test_Device'] == dev]
        
        # --- NEW: Draw a reference diagonal identity line (y = x) ---
        ax.plot([axis_min, axis_max], [axis_min, axis_max], color='gray', linestyle=':', alpha=0.5, zorder=1)
        
        for strategy, marker in marker_map.items():
            subset_strat = subset_dev[subset_dev['Strategy'] == strategy]
            
            for model_name in models:
                subset = subset_strat[subset_strat['Model'] == model_name]
                if not subset.empty:
                    label = f"{model_name}" if strategy == "All_MACs" and idx == 0 else "_nolegend_"
                    ax.scatter(subset['Macro_Recall'], subset['Macro_Precision'], 
                               color=color_map[model_name], marker=marker, s=150, 
                               label=label, edgecolors='black', alpha=0.9, zorder=2)

        ax.set_title(f"Test Device: {dev}", fontsize=14, fontweight='bold')
        ax.set_xlabel("Macro Recall", fontsize=12)
        ax.set_ylabel("Macro Precision", fontsize=12) # Kept label on both since they're standalone squares now
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # --- NEW: Lock symmetric limits and force a perfect square aspect ratio ---
        ax.set_xlim(axis_min, axis_max)
        ax.set_ylim(axis_min, axis_max)
        ax.set_aspect('equal', adjustable='box')

    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='All MACs', markerfacecolor='gray', markersize=12),
                       Line2D([0], [0], marker='*', color='w', label='Filtered MACs', markerfacecolor='gray', markersize=15),
                       Line2D([0], [0], color='gray', linestyle=':', label='Diagonal (P = R)')] # Added to legend
    for m in models:
        legend_elements.append(Line2D([0], [0], marker='s', color='w', label=m, markerfacecolor=color_map[m], markersize=10))

    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.14), ncol=4, fontsize=11)
    
    plt.tight_layout()
    fig.subplots_adjust(top=0.90) # Adjusted downward slightly to fit the square plots + high legend cleanly
    
    plot_path = os.path.join(OUTPUT_DIR, "precision_recall_comparison.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] Precision-Recall Plot exported to {plot_path}")

    # 2. Confusion Matrix Plot
    if os.path.exists(CM_PREDS_CSV):
        cm_df = pd.read_csv(CM_PREDS_CSV)
        labels = sorted(np.unique(np.concatenate((cm_df['y_true'], cm_df['y_pred']))))
        
        cm = confusion_matrix(cm_df['y_true'], cm_df['y_pred'], labels=labels, normalize='true')
        
        fig_cm, ax_cm = plt.subplots(figsize=(10, 8))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(ax=ax_cm, cmap='Blues', xticks_rotation=45, colorbar=True, values_format='.2f')
        
        plt.title("Normalized Confusion Matrix: TLoc Model\n(Device V2, Filtered MACs)", fontsize=16, fontweight='bold')
        plt.xlabel("Predicted Zone", fontsize=12)
        plt.ylabel("True Zone", fontsize=12)
        plt.tight_layout()
        
        cm_plot_path = os.path.join(OUTPUT_DIR, "confusion_matrix_tloc_v2_filtered.png")
        plt.savefig(cm_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[SAVED] Confusion Matrix exported to {cm_plot_path}")
    else:
        print("[WARNING] Could not find TLoc predictions for Confusion Matrix.")

# --- 6. EXPORT HYPERPARAMS TO MARKDOWN ---
def export_hyperparams():
    md_content = """# Appendix: Model Hyperparameters

| Model | Hyperparameters |
| :--- | :--- |
| **XGBoost** | `n_estimators=100`, `max_depth=6`, `random_state=42`, `booster='gbtree'` |
| **WKNN_Pos_Euclid** | `n_neighbors=5`, `weights='distance'`, `metric='euclidean'`, `Representation=Positive` |
| **WKNN_Pow_Euclid** | `n_neighbors=5`, `weights='distance'`, `metric='euclidean'`, `Representation=Powered (e)` |
| **WKNN_Pos_Sorensen** | `n_neighbors=5`, `weights='distance'`, `metric='braycurtis'` (Sorensen), `Representation=Positive` |
| **WKNN_Pow_Sorensen** | `n_neighbors=5`, `weights='distance'`, `metric='braycurtis'` (Sorensen), `Representation=Powered (e)` |
| **TLoc (RBF)** | Default configuration defined inside python script class |
"""
    md_path = os.path.join(OUTPUT_DIR, "hyperparameters.md")
    with open(md_path, "w") as f:
        f.write(md_content)
    print(f"[SAVED] Hyperparameters table exported to {md_path}")

if __name__ == "__main__":
    generate_plots()
    export_hyperparams()
    print("\n--- ML Training and Evaluation Pipeline Finished ---")