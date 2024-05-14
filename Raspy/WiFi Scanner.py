# sudo apt install hwloc
# sudo apt install aircrack-ng
# sudo airmon-ng check kill
# airmon-ng start wlan0 (or other interface as wlp3s0)

# Todo: nao parar as outras interfaces
# todo: varrer todos os canais ao mesmo tempo
# todo: pegar 5ghz
# todo: ver wpa supplicant e nmcli configs para evitar desligar tudo


import scapy.all as scapy
from threading import Thread
import pandas
import time
import os

# initialize the networks dataframe that will contain all access points nearby
networks = pandas.DataFrame(columns=["BSSID", "SSID", "dBm_Signal", "Channel", "Crypto"])
# set the index BSSID (MAC address of the AP)
networks.set_index("BSSID", inplace=True)

def callback(packet):
    if packet.haslayer(scapy.Dot11Beacon):
        # extract the MAC address of the network
        bssid = packet[scapy.Dot11].addr2
        # get the name of it
        ssid = packet[scapy.Dot11Elt].info.decode()
        try:
            dbm_signal = packet.dBm_AntSignal
        except:
            dbm_signal = "N/A"
        # extract network stats
        stats = packet[scapy.Dot11Beacon].network_stats()
        # get the channel of the AP
        channel = stats.get("channel")
        # get the crypto
        crypto = stats.get("crypto")
        networks.loc[bssid] = (ssid, dbm_signal, channel, crypto)


def print_all():
    while True:
        os.system("clear")
        print(networks)
        time.sleep(0.5)


def change_channel():
    ch = 1
    while True:
        os.system(f"iwconfig {interface} channel {ch}")
        # switch channel from 1 to 13 each 0.5s
        ch = ch % 13 + 1
        time.sleep(0.5)


if __name__ == "__main__":
    # interface name, check using iwconfig
    interface = "wlp3s0mon"
    # start the thread that prints all the networks
    printer = Thread(target=print_all)
    printer.daemon = True
    printer.start()
    # start the channel changer
    # channel_changer = Thread(target=change_channel)
    # channel_changer.daemon = True
    # channel_changer.start()
    # start sniffing
    scapy.sniff(prn=callback, iface=interface)
