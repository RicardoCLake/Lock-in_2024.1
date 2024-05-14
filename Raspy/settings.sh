#!/bin/sh

# Install driver for Wifi adapter (tp-link ac1300 Acher T3U Plus)
sudo git clone "https://github.com/RinCat/RTL88x2BU-Linux-Driver.git" /usr/src/rtl88x2bu-git
sudo sed -i 's/PACKAGE_VERSION="@PKGVER@"/PACKAGE_VERSION="git"/g' /usr/src/rtl88x2bu-git/dkms.conf
sudo dkms add -m rtl88x2bu -v git
sudo dkms autoinstall
#sudo reboot # reboot is required to load the driver

# copies
scp -r pi@192.168.33.222:/home/pi/Lockin/Raspy/* "/home/rick/Dropbox/disciplininhas/Pole Location/Lock-in_2024.1/Raspy"
scp -r "/home/rick/Dropbox/disciplininhas/Pole Location/Lock-in_2024.1/Raspy" pi@192.168.33.222:/home/pi/Lockin/

wpa_cli
> scan
> scan_results
> interface
> set ignore_old_scan_res 1
> ap_scan 1
> set autoscan periodic:10
> scan_interval 10

https://wiki.archlinux.org/title/Wpa_supplicant
https://android.googlesource.com/platform/external/wpa_supplicant_8/+/master/wpa_supplicant/wpa_supplicant.conf
https://w1.fi/cgit/hostap/plain/wpa_supplicant/wpa_supplicant.conf

nmcli device
set [ifname] ifname [autoconnect {yes | no}] [managed {yes | no}]
           Set device properties.
down ifname...
           Disconnect a device and prevent the device from automatically activating further connections without user/manual intervention. Note that
           disconnecting software devices may mean that the devices will disappear.

           If --wait option is not specified, the default timeout will be 10 seconds.
wifi rescan [ifname ifname] [ssid SSID...]
           Request that NetworkManager immediately re-scan for available access points. NetworkManager scans Wi-Fi networks periodically, but in some
           cases it can be useful to start scanning manually (e.g. after resuming the computer). By using ssid, it is possible to scan for a specific
           SSID, which is useful for APs with hidden SSIDs. You can provide multiple ssid parameters in order to scan more SSIDs.

           This command does not show the APs, use nmcli device wifi list for that.

nmcli device show p2p-dev-wlp3s0
GENERAL.DEVICE:                         p2p-dev-wlp3s0
GENERAL.TYPE:                           wifi-p2p
GENERAL.HWADDR:                         (desconhecido)
GENERAL.MTU:                            0
GENERAL.STATE:                          30 (desconectado)
GENERAL.CONNECTION:                     --
GENERAL.CON-PATH:                       --

nmcli device status
DEVICE          TYPE      STATE            CONNECTION        
wlp3s0          wifi      conectado        ViaRezo           
enp2s0          ethernet  conectado        Conexão cabeada 1 
p2p-dev-wlp3s0  wifi-p2p  desconectado     --                
lo              loopback  não gerenciável  --     