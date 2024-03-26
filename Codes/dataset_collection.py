import pywifi
import time
import numpy as np
import time
import pandas as pd
from datetime import datetime
import os

class DatasetCollector:
    """
    The room is usually divided in 5 points, and the last one is outside the room.
    If door=0 means that the door is not closed and door=1 means the door is closed.
    In the automatic mode, the next collection will only start when you press enter
    """

    def __init__(self, room, door, part_of_the_room, duration_of_the_collection, good_wifis=None, verbose=True):
        self.room = room
        self.door = door
        self.part_of_the_room = part_of_the_room
        self.duration_of_the_collection = duration_of_the_collection
        wifi = pywifi.PyWiFi()
        self.iface = wifi.interfaces()[0]
        self.all_scans = np.empty((0, 4))
        self.good_wifis = good_wifis
        if good_wifis == None:
            self.get_all_wifis = True
        self.t0 = time.time()
        self.verbose = verbose
        self.i = 0
        if not os.path.exists("data"):
            # Create a new directory because it does not exist
            os.makedirs("data")


    def save(self):
        df = pd.DataFrame(self.all_scans, columns=['ap_mac', 'ap_name', 'rssi', 'timestamp'])
        df["room"] = self.room
        df["door_status"] = self.door
        df["room_part"] = self.part_of_the_room
        df.to_csv(
            f"data/{self.room} - {self.part_of_the_room} - {self.door} - {datetime.now().strftime('%Y_%m_%d-%I_%M_%S_%p')}.csv")

    def scan(self):
        scaned = []
        t = time.time()
        self.iface.scan()
        time.sleep(1)
        results = self.iface.scan_results()
        for i in results:
            if self.get_all_wifis or i.ssid in self.good_wifis:
                bssid = i.bssid
                ssid = i.ssid
                signal = i.signal
                scaned.append([bssid, ssid, signal, t])

        scaned = np.array(scaned)
        try:
            _, uniques_index = np.unique(scaned[:, 0], return_index=True)
            scaned = scaned[uniques_index]
        except:
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
                wait = input(f"Get ready to collect door {door} and part {part} of your room!")
                self.part_of_the_room = part
                self.door = door
                self.collect()
                self.save()
                self.i = 0
                self.all_scans = np.empty((0, 4))

if __name__=="__main__":
    # good_wifis = ["Guest-CentraleSupelec", "eduroam", 'stop&go', 'CD91', 'fabrique2024']
    room = "LC413"
    door_input = 0
    part_input = 1
    duration = 50
    dc = DatasetCollector(room, door_input, part_input, duration)
    dc.automatic_collection()

