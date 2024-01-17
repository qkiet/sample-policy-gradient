# Introduction
A sample policy gradient method model that is used to solve CartPole game. Referenced code in [here](https://github.com/abhisheksuran/Reinforcement_Learning/blob/master/Reinforce_(PG)_ReUploaded.ipynb)

# Setup
Tested with WSL Ubuntu 22.04
1. Install NVIDIA CUDA Toolkit 12.2. Refer [here](https://docs.nvidia.com/cuda/wsl-user-guide/index.html) and [here](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_network) (remember to change from *12.3* to *12.2*). TL;DR:
```bash
$ sudo apt-key del 7fa2af80
$ wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
$ sudo dpkg -i cuda-keyring_1.1-1_all.deb
$ sudo apt-get update
$ sudo apt-get -y install cuda-toolkit-12-2
```
2. Install package from requirement.txt by command
```bash
$ pip install -r requirements.txt
```

# Usage
```bash
$ python
```