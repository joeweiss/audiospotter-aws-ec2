
# Update
sudo apt update -y

# Add pyenv dependencies.
sudo apt install -y build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev curl \
libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev

# Postgis dependency, possibly unneeded since Django and DB isn't running on this instance.
sudo apt-get install gdal-bin

# Install ffmpeg
sudo apt install ffmpeg -y

# Instal pyenv
curl https://pyenv.run | bash

echo 'export PYENV_ROOT="$HOME/.pyenv"' >> /home/ubuntu/.bashrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> /home/ubuntu/.bashrc
echo 'eval "$(pyenv init -)"' >> /home/ubuntu/.bashrc

# Run manually (rather than restart the shell as you normally would when installing pyenv)
export PYENV_ROOT="$HOME/.pyenv"
command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"

pyenv --version

# Install desired python3 version.
PYTHON_VERSION=3.10.10
pyenv install $PYTHON_VERSION
pyenv global $PYTHON_VERSION
python --version > /home/ubuntu/python_version.txt

# Create environment
VENV=birdnetlib-env
pyenv virtualenv $PYTHON_VERSION $VENV
pyenv activate $VENV

python -m pip install --upgrade pip
pip install -r requirements.txt

# To watch this process: tail -f /var/log/cloud-init-output.log

# Install cron.
crontab < cron.txt

# Set extraction directory.
mkdir extractions

# Start services
RUNNER_COUNT=4

for (( c=1; c<=$RUNNER_COUNT; c++ ))
do
    sudo cp /home/ubuntu/birdnetlib-aws-runner/runner.service /etc/systemd/system/runner_$c.service
    sudo systemctl enable runner_$c
    sudo systemctl daemon-reload
    sudo systemctl start runner_$c
done

# To follow along, do this: sudo journalctl -u runner_1 -f

# To see killed processes, do this: sudo dmesg | grep -i kill
