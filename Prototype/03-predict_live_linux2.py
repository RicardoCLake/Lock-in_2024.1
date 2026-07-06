import pandas as pd
import numpy as np
import os
import sys
import time
import glob
import re
import subprocess
import threading
import queue
import tkinter as tk
import scipy.stats
from scipy.special import gamma, hyp2f1
from sklearn.neighbors import KNeighborsClassifier
from collections import deque

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

# --- COLETA WI-FI VIA NMCLI ---
class NmcliWifiScanner:
    """
    Coleta de amostras Wi-Fi via NetworkManager (nmcli), substituindo o
    pywifi. No Linux com NetworkManager ativo, o pywifi frequentemente
    retorna resultados vazios ou presos em cache porque o NM já controla
    a interface e disputa com o wpa_supplicant que o pywifi tenta usar
    diretamente. Falar com o próprio NM via nmcli é mais estável.
    """

    _SPLIT_RE = re.compile(r'(?<!\\):')

    def __init__(self, iface=None, verbose=False):
        self.verbose = verbose
        self.iface = iface or self._detect_wifi_interface()
        if self.verbose:
            print(f"Usando interface: {self.iface}")

    def _detect_wifi_interface(self):
        try:
            output = subprocess.check_output(
                ["nmcli", "-t", "-f", "DEVICE,TYPE", "device", "status"],
                text=True
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Não foi possível listar dispositivos via nmcli: {e}")

        for line in output.strip().split("\n"):
            parts = line.split(":")
            if len(parts) >= 2 and parts[1] == "wifi":
                return parts[0]

        raise RuntimeError("Nenhuma interface wifi encontrada pelo nmcli.")

    def _unescape(self, value):
        return value.replace("\\:", ":")

    def _split_terse_line(self, line):
        raw_fields = self._SPLIT_RE.split(line)
        return [self._unescape(f) for f in raw_fields]

    def _request_rescan(self):
        result = subprocess.run(
            ["nmcli", "device", "wifi", "rescan", "ifname", self.iface],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True
        )
        if result.returncode != 0:
            stderr = result.stderr.strip()
            if "Scanning not allowed" in stderr or "not authorized" in stderr:
                if self.verbose:
                    print("ℹ️ NM recusou novo rescan (muito recente). Usando cache atual do NM...")
            else:
                if self.verbose:
                    print(f"⚠️ nmcli rescan retornou erro: {stderr or '(sem mensagem)'}")

    def _list_wifi(self):
        try:
            output = subprocess.check_output(
                ["nmcli", "-t", "-f", "BSSID,SSID,SIGNAL", "device", "wifi", "list", "ifname", self.iface],
                text=True,
                stderr=subprocess.DEVNULL
            )
        except subprocess.CalledProcessError:
            return []

        entries = []
        for line in output.strip().split("\n"):
            if not line:
                continue
            fields = self._split_terse_line(line)
            if len(fields) < 3:
                continue
            bssid, ssid, signal = fields[0], fields[1], fields[2]
            if not bssid:
                continue
            entries.append((bssid, ssid, signal))
        return entries

    def scan(self, mac_filter=None, max_retries=100):
        """
        Executa um scan com retry, igual ao usado na coleta do dataset:
        tenta de novo se vier vazio (NM ainda sem resposta) ou se o
        resultado for idêntico ao scan anterior (cache preso).
        Retorna um dict {bssid: signal} já filtrado por mac_filter, se dado.
        """
        retries = 0
        results = []

        while retries < max_retries:
            self._request_rescan()
            time.sleep(1)  # tempo estritamente necessário para o hardware escanear

            results = self._list_wifi()

            current_scan_data = {
                (bssid, signal) for bssid, ssid, signal in results
                if mac_filter is None or bssid in mac_filter
            }

            if not current_scan_data:
                if self.verbose:
                    print("⚠️ Retorno vazio (NM ainda sem resposta). Tentando de novo...")
                retries += 1
                time.sleep(0.5)
                continue

            if hasattr(self, 'last_scan_data') and current_scan_data == self.last_scan_data:
                if self.verbose:
                    print("♻️ Dados repetidos (cache detectado). Forçando novo scan...")
                retries += 1
                time.sleep(0.5)
                continue

            self.last_scan_data = current_scan_data
            break
        else:
            if self.verbose:
                print("❌ Excedeu o número máximo de tentativas sem obter um scan válido.")

        scan_dict = {}
        for bssid, ssid, signal in results:
            if mac_filter is None or bssid in mac_filter:
                try:
                    scan_dict[bssid] = int(signal)
                except ValueError:
                    continue
        return scan_dict


# --- INTERFACE GRÁFICA: DESTAQUE DA SUBREGIÃO DETECTADA ---
class PositionGUI:
    """
    Desenha a sala dividida em 4 quadrantes (1-4) + a região 5 (fora / corredor,
    representada pela barra no topo), no mesmo layout mostrado no protocolo:

        [ 5 (fora / corredor) ]
        [  1  |  2  ]
        [  4  |  3  ]

    O método highlight(regiao) pinta apenas o quadrante detectado.
    """

    # (x0, y0, x1, y1) de cada região dentro do canvas
    REGIONS = {
        1: (30, 110, 195, 260),
        2: (205, 110, 370, 260),
        4: (30, 270, 195, 420),
        3: (205, 270, 370, 420),
        5: (30, 20, 370, 80),
    }

    COLOR_IDLE = "#ffffff"
    COLOR_IDLE_5 = "#f2f2f2"
    COLOR_HIGHLIGHT = "#ffd54f"
    TEXT_IDLE = "#0b3d91"
    TEXT_HIGHLIGHT = "#8a5600"

    def __init__(self, master):
        self.master = master
        master.title("Predição de posição ao vivo")
        master.resizable(False, False)

        self.canvas = tk.Canvas(master, width=400, height=450, bg="white", highlightthickness=0)
        self.canvas.pack(padx=10, pady=(10, 0))

        # Contorno da sala (quadrantes 1-4)
        self.canvas.create_rectangle(30, 110, 370, 420, outline="#333333", width=3)
        self.canvas.create_line(200, 110, 200, 420, fill="#333333", width=3)
        self.canvas.create_line(30, 265, 370, 265, fill="#333333", width=3)

        self.rects = {}
        self.texts = {}
        for region, (x0, y0, x1, y1) in self.REGIONS.items():
            fill = self.COLOR_IDLE_5 if region == 5 else self.COLOR_IDLE
            rect = self.canvas.create_rectangle(x0, y0, x1, y1, fill=fill, outline="")
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            text = self.canvas.create_text(
                cx, cy, text=str(region), font=("Helvetica", 34, "bold"), fill=self.TEXT_IDLE
            )
            self.rects[region] = rect
            self.texts[region] = text

        self.info_label = tk.Label(master, text="Aguardando amostras...", font=("Helvetica", 12))
        self.info_label.pack(pady=10)

    def highlight(self, region, info_text=""):
        for r, rect_id in self.rects.items():
            if r == region:
                self.canvas.itemconfig(rect_id, fill=self.COLOR_HIGHLIGHT)
                self.canvas.itemconfig(self.texts[r], fill=self.TEXT_HIGHLIGHT)
            else:
                base_fill = self.COLOR_IDLE_5 if r == 5 else self.COLOR_IDLE
                self.canvas.itemconfig(rect_id, fill=base_fill)
                self.canvas.itemconfig(self.texts[r], fill=self.TEXT_IDLE)
        if info_text:
            self.info_label.config(text=info_text)


# --- LÓGICA PRINCIPAL AO VIVO ---
class LivePredictor:
    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.alpha = 1#0.84
        self.history = deque(maxlen=1)
        
        # Inicializa o scanner Wi-Fi via nmcli
        self.scanner = NmcliWifiScanner(verbose=False)
        
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
        scan_dict = self.scanner.scan(mac_filter=self.mac_cols)
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
        self.root = tk.Tk()
        self.gui = PositionGUI(self.root)
        self.result_queue = queue.Queue()
        self.stop_event = threading.Event()

        self.worker_thread = threading.Thread(target=self._prediction_loop, daemon=True)
        self.worker_thread.start()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._poll_queue()
        self.root.mainloop()

    def _on_close(self):
        self.stop_event.set()
        self.root.destroy()

    def _poll_queue(self):
        try:
            while True:
                region, info_text = self.result_queue.get_nowait()
                self.gui.highlight(region, info_text)
        except queue.Empty:
            pass
        if not self.stop_event.is_set():
            self.root.after(100, self._poll_queue)

    def _prediction_loop(self):
        count = 1
        while not self.stop_event.is_set():
            try:
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
                ages = np.arange(n_samples - 1, -1, -1)
                weights = self.alpha ** ages

                weighted_sum = np.zeros(len(self.classes))
                for i, prob in enumerate(self.history):
                    weighted_sum += prob * weights[i]

                final_probs = weighted_sum / np.sum(weights)

                best_idx = np.argmax(final_probs)
                predicted_position = self.classes[best_idx]
                confidence = final_probs[best_idx] * 100

                # A subregião é o que vem depois do último "_" na classe (ex: "QQQ_3" -> 3)
                try:
                    region = int(str(predicted_position).split("_")[-1])
                except (ValueError, IndexError):
                    region = None

                info_text = (
                    f"Amostra {count:03d} | Janela {n_samples:02d}/{n_samples} | "
                    f"Posição: {predicted_position} | Confiança: {confidence:.1f}%"
                )
                self.result_queue.put((region, info_text))

                count += 1
            except Exception as e:
                # Não deixa a thread morrer silenciosamente em caso de erro pontual de scan
                self.result_queue.put((None, f"⚠️ Erro na amostra {count:03d}: {e}"))
                time.sleep(1)

if __name__ == "__main__":
    folder = input("Digite o caminho da pasta onde estão os arquivos do modelo (ex: data_merged): ")
    if not os.path.exists(folder):
        print("Pasta não encontrada.")
    else:
        predictor = LivePredictor(folder)
        predictor.run()