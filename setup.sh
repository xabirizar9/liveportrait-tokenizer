echo "Installing zsh and Oh My Zsh..."
sudo apt-get update
sudo apt-get install -y zsh ffmpeg
sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended


# Set variables
MOUNTING_PATH="/mnt/disks/celebv-hq"

# Create directory for CelebV-HQ dataset
echo "Creating directory for CelebV-HQ dataset..."
sudo mkdir -p $MOUNTING_PATH

# Find and mount the CelebV-HQ disk
echo "Finding and mounting CelebV-HQ disk..."
DISK_ID=$(ls -l /dev/disk/by-id/google* | grep "google-celebv-hq" | awk '{print $NF}' | sed 's/\.\.\/\.\.\///')
if [[ $DISK_ID == *"sda"* ]]; then
  DISK_PATH="/dev/sda"
elif [[ $DISK_ID == *"sdb"* ]]; then
  DISK_PATH="/dev/sdb"
else
  echo "Error: Could not find the CelebV-HQ disk"
  exit 1
fi

echo "Mounting disk $DISK_PATH to $MOUNTING_PATH..."
sudo mount -o discard,defaults $DISK_PATH $MOUNTING_PATH

# Create symbolic link to dataset
echo "Creating symbolic link to dataset..."
ln -s $MOUNTING_PATH/CelebV-HQ ./dataset

sudo chmod -R 777 $MOUNTING_PATH/

pip install uv

uv sync

.venv/bin/python -m huggingface_hub.commands.huggingface_cli download KwaiVGI/LivePortrait --local-dir pretrained_weights --exclude "*.git*" "README.md" "docs"
