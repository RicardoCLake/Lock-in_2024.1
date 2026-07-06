import subprocess
import shlex
import re
import time
import numpy as np
import pandas as pd
from datetime import datetime
import os


class DatasetCollector:
    """
    The room is usually divided in 5 points, and the last one is outside the room.
    If door=0 means that the door is not closed and door=1 means the door is closed.
    In the automatic mode, the next collection will only start when you press enter

    Esta versão usa `nmcli` (NetworkManager) em vez de pywifi, pois no Linux
    (especialmente com NetworkManager ativo) o pywifi frequentemente retorna
    resultados vazios ou presos em cache, já que o NM disputa o controle da
    interface com o wpa_supplicant que o pywifi tenta usar diretamente.
    """

    # Regex para dividir uma linha do nmcli -t respeitando os ':' escapados (\:)
    _SPLIT_RE = re.compile(r'(?<!\\):')

    def __init__(self, room, door, part_of_the_room, duration_of_the_collection,
                 good_wifis=None, iface=None, verbose=True):
        self.room = room
        self.door = door
        self.part_of_the_room = part_of_the_room
        self.duration_of_the_collection = duration_of_the_collection
        self.iface = iface or self._detect_wifi_interface()
        self.all_scans = np.empty((0, 4))
        self.good_wifis = good_wifis
        self.get_all_wifis = good_wifis is None
        self.t0 = time.time()
        self.verbose = verbose
        self.i = 0
        if not os.path.exists("data"):
            os.makedirs("data")

        if self.verbose:
            print(f"Usando interface: {self.iface}")

    # ------------------------------------------------------------------ #
    # Utilitários de baixo nível para falar com o nmcli
    # ------------------------------------------------------------------ #

    def _detect_wifi_interface(self):
        """Descobre automaticamente a interface wifi via nmcli."""
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
        """Desfaz o escape de ':' que o nmcli -t aplica (\\: -> :)."""
        return value.replace("\\:", ":")

    def _split_terse_line(self, line):
        """Divide uma linha do nmcli -t respeitando ':' escapados."""
        raw_fields = self._SPLIT_RE.split(line)
        return [self._unescape(f) for f in raw_fields]

    def _request_rescan(self):
        """
        Pede ao NetworkManager para re-escanear. O NM às vezes recusa com
        'Error: Scanning not allowed immediately following previous scan.'
        quando chamado com muita frequência — tratamos isso como um caso
        não-fatal e apenas seguimos para ler a lista já em cache do NM.
        """
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
        """
        Lê a lista atual de redes vista pelo NM. Retorna lista de tuplas
        (bssid, ssid, signal). Pode vir vazia se o NM ainda não tiver
        respondido ao rescan.
        """
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
                # linha malformada / incompleta, ignora
                continue
            bssid, ssid, signal = fields[0], fields[1], fields[2]
            if not bssid:
                continue
            entries.append((bssid, ssid, signal))
        return entries

    # ------------------------------------------------------------------ #
    # API pública (mantida igual à versão anterior)
    # ------------------------------------------------------------------ #

    def save(self):
        df = pd.DataFrame(self.all_scans, columns=['ap_mac', 'ap_name', 'rssi', 'timestamp'])
        df["room"] = self.room
        df["door_status"] = self.door
        df["room_part"] = self.part_of_the_room
        df.to_csv(
            f"data/{self.room}-{self.part_of_the_room}-{self.door}-{datetime.now().strftime('%Y_%m_%d-%I_%M_%S_%p')}.csv")

    def scan(self):
        max_retries = 100
        retries = 0
        results = []

        while retries < max_retries:
            # 1. Pede ao NetworkManager para escanear fisicamente
            self._request_rescan()
            time.sleep(2.5)  # Tempo necessário para varrer os canais

            # 2. Pega os resultados que o NM já processou
            results = self._list_wifi()

            # 3. Cria um conjunto (bssid, signal) para comparar com o scan anterior
            current_scan_data = {
                (bssid, signal) for bssid, ssid, signal in results
                if self.get_all_wifis or bssid in self.good_wifis
            }

            # Verificação A: Está vazio?
            if not current_scan_data:
                if self.verbose:
                    print("⚠️ Retorno vazio (NM ainda sem resposta). Tentando de novo...")
                retries += 1
                time.sleep(1)
                continue

            # Verificação B: É idêntico ao último scan? (Preso em cache)
            if hasattr(self, 'last_scan_data') and current_scan_data == self.last_scan_data:
                if self.verbose:
                    print("♻️ Dados repetidos (cache detectado). Forçando novo scan...")
                retries += 1
                time.sleep(1)
                continue

            # Dados frescos! Salvamos para comparar na próxima iteração
            self.last_scan_data = current_scan_data
            break  # Sai do loop while
        else:
            if self.verbose:
                print("❌ Excedeu o número máximo de tentativas sem obter um scan válido.")

        # Processamento final (igual à versão anterior)
        scaned = []
        t = time.time() - self.t0
        for bssid, ssid, signal in results:
            if self.get_all_wifis or (bssid in self.good_wifis):
                scaned.append([bssid, ssid, signal, t])

        scaned = np.array(scaned)
        try:
            _, uniques_index = np.unique(scaned[:, 0], return_index=True)
            scaned = scaned[uniques_index]
        except Exception:
            scaned = np.empty((0, 4))

        if self.verbose:
            print(f"Iteration: {self.i}")
            print(f"APs detected: {scaned.shape[0]}")

        return scaned

    def collect(self):
        while self.i < self.duration_of_the_collection:
            scaned = self.scan()
            self.all_scans = np.concatenate((self.all_scans, scaned))
            self.i += 1

    def automatic_collection(self):
        doors = [0, 1]
        parts = [1, 2, 3, 4, 5]
        for door in doors:
            for part in parts:
                input(f"Get ready to collect door {door} and part {part} of your room!")
                self.part_of_the_room = part
                self.door = door
                self.collect()
                self.save()
                self.i = 0
                self.all_scans = np.empty((0, 4))

if __name__ == "__main__":
    # good_wifis = ["Guest-CentraleeSupelec", "eduroam", 'stop&go', 'CD91', 'fabrique2024']
    room = ""
    while room == "":
        room = input("Give-me the room name:")
    door_input = 0
    part_input = 1
    duration = 20
    dc = DatasetCollector(room, door_input, part_input, duration)
    dc.automatic_collection()
