import pywifi
import time
import numpy as np
import pandas as pd
from datetime import datetime
import os


class RaspyDatasetCollector:
    def __init__(self, room:str, wifiInterface, scanningWaitingTime, expirationAge, good_wifis=None, verbose=True):
        self.room = room
        self.iface_object = wifiInterface
        self.iface_name = wifiInterface.name()
        self.scanningWaitingTime = scanningWaitingTime
        self.good_wifis = good_wifis
        self.verbose = verbose
        self.get_all_wifis = (good_wifis == None)
        
        self.all_scans = np.empty((0, 4))
        self.numOfSamples = 0
        
        if not os.path.exists("data"):
            # Create a new directory because it does not exist
            os.makedirs("data")
        
        if os.name == 'posix':  # Unix-like system
            final_string = " > /dev/null 2>&1"
        elif os.name == 'nt':   # Windows
            final_string = " > nul 2>&1"
        
        os.system("nmcli device down {}".format(self.iface_name) + final_string)
        os.system("wpa_cli -i {} set bss_expiration_age {}".format(self.iface_name,int(expirationAge)) + final_string)
        os.system("wpa_cli -i {} set ignore_old_scan_res 1".format(self.iface_name) + final_string)
        os.system("wpa_cli -i {} scan".format(self.iface_name) + final_string) 
        time.sleep(self.scanningWaitingTime*10)
        os.system("wpa_cli -i {} scan_results".format(self.iface_name) + final_string) # to clean output 

    def _toDataFrame(self): 
        df = pd.DataFrame(self.all_scans, columns=['ap_mac', 'ap_name', 'rssi', 'timestamp'])
        df["room"] = self.room
        return df
    
    def save(self):
        df = self._toDataFrame()
        df.to_csv(
            "data/D_{}_{}_{}_{}.csv".format(self.room,self.iface_name,self.numOfSamples,datetime.now().strftime('%Y_%m_%d-%I_%M_%S_%p')))
    
    def print(self):
        df = self._toDataFrame()
        print("DATA {} {} : {} samples at {}".format(self.room,self.iface_name,self.numOfSamples,datetime.now().strftime('%Y_%m_%d-%I_%M_%S_%p')))
        print(df)

    def _scan(self):
        scaned = []
        timestamp0 = time.time()
        self.iface_object.scan()
        time.sleep(self.scanningWaitingTime)
        results = self.iface_object.scan_results()
        timestamp = time.time()
        # while still no new results
        while(len(results) == 0):
            time.sleep(self.scanningWaitingTime/100)
            results = self.iface_object.scan_results() 
            timestamp = time.time()
            if timestamp - timestamp0 >= 15*self.scanningWaitingTime:
                print("SCANNING TIMEOUT")
                exit()
            
        # Noting results
        for router in results:
            if self.get_all_wifis or router.ssid in self.good_wifis:
                bssid = router.bssid #mac address
                ssid = router.ssid #name
                signal = router.signal #rssid
                scaned.append([bssid, ssid, signal, timestamp])
        scaned = np.array(scaned)
        try:
            _, uniques_index = np.unique(scaned[:, 0], return_index=True)
            scaned = scaned[uniques_index]
        except:
            scaned = np.empty((0, 4))
        return scaned, timestamp-timestamp0

    def collect(self):
        
        scaned, delay = self._scan()
        self.all_scans = np.concatenate((self.all_scans, scaned))
        self.numOfSamples += 1
        if self.verbose:
            print("Iteration: {}".format(self.numOfSamples))
            print("  Num of detected APs: {}".format(scaned.shape[0]))
            print("  Scanning delay: {}".format(delay))
          
    def getData(self):
        return self._toDataFrame()
    
    def getNumOfSamples(self):
        return self.numOfSamples
            
    def erase(self):
        self.all_scans = np.empty((0, 4))
        self.numOfSamples = 0
        if self.verbose:
            print("-- Samples erased")

class RaspyDataFormatting:
    def __init__(self, room:str, iface_name:str, samples):
        self.iface_name = iface_name
        self.room = room
        self.DB = samples
        self.MACmap = {}
        self.pivotDB = pd.DataFrame()
        
        macs = samples["ap_mac"].unique()
        for mac in macs:
            self.MACmap[mac] = self.DB.loc[self.DB['ap_mac'] == mac, 'ap_name'].values[0]
        self.pivotDB = self.DB.pivot(index='timestamp', columns='ap_mac', values='rssi')
    
    def getMACMapping(self):
        return self.MACmap
    
    def getFormattedData(self):
        return self.pivotDB

    def save(self):
        now = datetime.now()
        numOfSamples = self.pivotDB.shape[0]
        self.pivotDB.to_csv(
            "data/P_{}_{}_{}_{}.csv".format(self.room,self.iface_name,numOfSamples,now.strftime('%Y_%m_%d-%I_%M_%S_%p')))
        
        keys = list(self.MACmap.keys())
        values = list(self.MACmap.values())
        df = pd.DataFrame(keys, columns=["ap_mac"])
        df["ap_name"] = values
        df.to_csv(
            "data/M_{}_{}_{}_{}.csv".format(self.room,self.iface_name,numOfSamples,now.strftime('%Y_%m_%d-%I_%M_%S_%p')))
        
    
if __name__=="__main__":
    # good_wifis = ["Guest-CentraleSupelec", "eduroam", 'stop&go', 'CD91', 'fabrique2024']
    room = input("Enter the room name: ")
    collectingInterval = 7 # seconds
    expirationAge = 5 # seconds (int)
    collectionWaitingTime = 2 # seconds
    num_of_samples = 100
    
    # Initialization
    wifi = pywifi.PyWiFi()
    num_of_interfaces = len(wifi.interfaces())
    print("Number of interfaces: ", num_of_interfaces)
    print("Interfaces: ", end="")
    for i, inter in enumerate(wifi.interfaces()):
        print(inter.name() + " (" + str(i) + "), ", end="")
    print("")
    
    chosen = int(input("Index of the chosen interface: "))    
    print("")   
    dc = RaspyDatasetCollector(room, wifi.interfaces()[chosen], collectionWaitingTime, expirationAge, verbose=True)
        
    # Scanning
    for i in range(num_of_samples):
        time1 = time.time()
        dc.collect()
        time2 = time.time()
        time.sleep(collectingInterval-(time2-time1))
    
    data = dc.getData()
    timestamples = np.unique(data["timestamp"]).astype(np.float64)
    print("\nSamples intervals:", np.diff(timestamples),"\n")
     
    # Saving
    dc.save()
    form = RaspyDataFormatting(room, wifi.interfaces()[chosen].name(), data)
    form.save()
    #os.system('rclone copy "/home/pi/Lockin/Raspy/data" "gdrive:Raspi Data"')

    
    
    
