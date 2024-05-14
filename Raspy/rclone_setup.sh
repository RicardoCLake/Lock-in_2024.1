# Install rclone
sudo apt-get update
sudo apt install rclone

# Setup Google Drive as remote
rclone config

# Configuration steps:
# - Select new remote
# - Enter name, e.g. 'gdrive'
# - Select Google Drive as storage type
# - Leave application client id empty
# - Leave application secret empty
# - Select manual config
# - Copy link into browser and authorize rclone for Google Drive
# - Copy verification code from browser and paste into console
# - Verify that the configuration is correct
# - Quit

# Actually sync something
rclone copy /home/pi/data/db.sqlite3 "gdrive:backups"

# Setup cron job to run daily backups
crontab -e

# Add new line to cron configuration, e.g. to run the above sync each day at midnight add the following line and save:
# 0 0 * * * rclone copy /home/pi/data/db.sqlite3 "gdrive:backups"