import pandas as pd
import numpy as np
import os
import sys
import time
import glob
import scipy.stats
from scipy.special import gamma, hyp2f1
from sklearn.neighbors import KNeighborsClassifier
import pywifi
from collections import deque
import logging

# Desativa os logs verbosos do pywifi para não poluir o terminal
logging.getLogger("pywifi").setLevel(logging.WARNING)

# --- CLASSE TLOC ADAPTADA ---
class TLoc:
    def __init__(self, train_data: pd.DataFrame, target_col="position"):
        self.train_data = train_data
        self.target_col = target_col
        self.aps = [col for col in train_data.columns if col != target_col]
        
        if len(self.aps) == 1:
            self.max_power = int(self.train_data[self.aps].max())
        else:
            self.max_power = int(self.train_data[self.aps].max().max())

        self.spaces = sorted(list(self.train_data[self.target_col].unique()))
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

    def pred_proba(self, X_test):
        all_probs = []
        min_prob = self.eps * np.ones(len(self.spaces))

        for _, test_sample in X_test.iterrows():
            distribution = np.ones(len(self.spaces))
            for router in self.aps:
                power = int(test_sample[router])
                try:
                    prob = self.power_probability_masks[router][power]
                except KeyError:
                    prob = min_prob
                prob = np.maximum(prob, min_prob)
                distribution = distribution * prob

            prob_sum = distribution.sum()
            normalized_probs = distribution / prob_sum if prob_sum > 0 else distribution
            all_probs.append(normalized_probs)
            
        return np.array(all_probs)

# --- FUNÇÕES DE PRÉ-PROCESSAMENTO ---
def apply_positive_repr(df, mac_cols):
    df_pos = df.copy()
    
    # Identifica quais colunas de metadados existem no DataFrame atual (ex: 'position' no treino)
    meta_cols_expected = ['timestamp', 'room', 'device_id', 'door_status', 'room_part', 'position']
    meta_cols_present = [col for col in meta_cols_expected if col in df_pos.columns]
    
    # O reindex agora preserva os metadados existentes E garante a ordem/presença exata dos MACs
    df_pos = df_pos.reindex(columns=meta_cols_present + mac_cols)
    
    # Aplica a transformação de potência APENAS nas colunas de MAC
    df_pos[mac_cols] = df_pos[mac_cols].fillna(-100.0)
    df_pos[mac_cols] = df_pos[mac_cols] + 100.0
    df_pos[mac_cols] = df_pos[mac_cols].clip(lower=0)
    
    return df_pos

def apply_powed_repr(df_pos, mac_cols):
    df_pow = df_pos.copy()
    df_pow[mac_cols] = (df_pow[mac_cols] / 100.0) ** np.e
    return df_pow

# --- LÓGICA PRINCIPAL AO VIVO ---
class LivePredictor:
    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.alpha = 0.84
        self.history = deque(maxlen=60)
        
        # Inicializa Wi-Fi
        wifi = pywifi.PyWiFi()
        if not wifi.interfaces():
            print("Erro: Nenhuma interface Wi-Fi encontrada!")
            sys.exit(1)
        self.iface = wifi.interfaces()[0]
        
        self.load_and_train()
        
    def load_and_train(self):
        print("\n> Carregando dados e treinando modelos. Aguarde...")
        all_files = glob.glob(os.path.join(self.data_folder, "*.csv"))
        if not all_files:
            print(f"Erro: Nenhum arquivo CSV encontrado em {self.data_folder}")
            sys.exit(1)
            
        df_list = [pd.read_csv(f, index_col=0) if 'Unnamed: 0' in pd.read_csv(f, nrows=0).columns else pd.read_csv(f) for f in all_files]
        df_train = pd.concat(df_list, ignore_index=True)
        
        # Criar a coluna target (Sala_Parte)
        df_train['position'] = df_train['room'].astype(str) + "_" + df_train['room_part'].astype(str)
        self.classes = sorted(list(df_train['position'].unique()))
        
        # Isolar MACs
        meta_cols = ['timestamp', 'room', 'device_id', 'door_status', 'room_part', 'position']
        self.mac_cols = [c for c in df_train.columns if c not in meta_cols]
        
        # Preparações para os modelos
        train_pos = apply_positive_repr(df_train, self.mac_cols)
        train_pow = apply_powed_repr(train_pos, self.mac_cols)
        
        # Treino TLoc
        print("> Treinando TLoc...")
        self.tloc = TLoc(train_pos[self.mac_cols + ['position']], target_col='position')
        self.tloc.train()
        
        # Treino WKNN (Sorensen / Bray-Curtis)
        print("> Treinando WKNN...")
        self.knn = KNeighborsClassifier(n_neighbors=5, weights='distance', metric='braycurtis')
        self.knn.fit(train_pow[self.mac_cols], train_pow['position'])
        
        # Alinhamento de classes do WKNN com nossa lista global
        self.knn_class_indices = [list(self.knn.classes_).index(c) for c in self.classes]
        
        print("\n> Modelos treinados! Iniciando predição ao vivo...\n")
        
    def scan_wifi(self):
        self.iface.scan()
        time.sleep(1) # Tempo estritamente necessário para o hardware escanear
        results = self.iface.scan_results()
        
        scan_dict = {i.bssid: i.signal for i in results if i.bssid in self.mac_cols}
        return pd.DataFrame([scan_dict])

    def align_probs(self, probs, source_classes):
        """Garante que os arrays de probabilidade fiquem na mesma ordem para a soma"""
        aligned = np.zeros(len(self.classes))
        for i, c in enumerate(self.classes):
            if c in source_classes:
                idx = list(source_classes).index(c)
                aligned[i] = probs[0][idx]
        return aligned

    def run(self):
        count = 1
        # Limpa a tela uma vez antes do loop
        os.system('cls' if os.name == 'nt' else 'clear')
        print("="*50)
        print(" PREDIÇÃO DE POSIÇÃO AO VIVO (TLoc + WKNN)".center(50))
        print("="*50)
        
        try:
            while True:
                # 1. Coleta e formata a amostra
                df_live = self.scan_wifi()
                df_pos = apply_positive_repr(df_live, self.mac_cols)
                df_pow = apply_powed_repr(df_pos, self.mac_cols)
                
                # 2. Predição dos modelos (Soft Voting)
                tloc_probs_raw = self.tloc.pred_proba(df_pos)
                knn_probs_raw = self.knn.predict_proba(df_pow)
                
                tloc_probs = self.align_probs(tloc_probs_raw, self.tloc.spaces)
                knn_probs = self.align_probs(knn_probs_raw, self.knn.classes_)
                
                # Média simples entre TLoc e WKNN para a amostra ATUAL
                current_ensemble_prob = (tloc_probs + knn_probs) / 2.0
                
                # 3. Adiciona ao histórico (deque empurra o mais antigo pra fora se passar de 60)
                self.history.append(current_ensemble_prob)
                
                # 4. Decaimento Exponencial Temporal
                n_samples = len(self.history)
                # O mais recente tem idade 0 (peso 0.84^0 = 1). O mais antigo tem idade N-1.
                ages = np.arange(n_samples - 1, -1, -1) 
                weights = self.alpha ** ages
                
                weighted_sum = np.zeros(len(self.classes))
                for i, prob in enumerate(self.history):
                    weighted_sum += prob * weights[i]
                
                # Normaliza para exibir a % de confiança
                final_probs = weighted_sum / np.sum(weights)
                
                best_idx = np.argmax(final_probs)
                predicted_position = self.classes[best_idx]
                confidence = final_probs[best_idx] * 100
                
                # 5. Print Inteligente (Sobrescreve a mesma linha)
                # \033[K apaga o resto da linha, \r volta para o começo
                sys.stdout.write(f"\r\033[K[Amostra {count:03d} | Janela: {n_samples:02d}/60] >> Posição: {predicted_position} ") #(Confiança: {confidence:.1f}%)
                sys.stdout.flush()
                
                count += 1
                
        except KeyboardInterrupt:
            print("\n\n> Teste finalizado pelo usuário!")

if __name__ == "__main__":
    folder = input("Digite o caminho da pasta onde estão os arquivos do modelo (ex: data_merged): ")
    if not os.path.exists(folder):
        print("Pasta não encontrada.")
    else:
        predictor = LivePredictor(folder)
        predictor.run()