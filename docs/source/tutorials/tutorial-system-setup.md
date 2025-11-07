# System Setup Tutorial
```{index} triple: System Setup; Tutorial; Ubuntu
```
```{raw} html
<span class="index-entry">System Setup</span>
<span class="index-entry">Tutorial</span>
<span class="index-entry">Ubuntu</span>
```

```{important}
The following instructions are tailored for **Ubuntu 24.04 LTS**. If you're using a different version, please adjust accordingly.
```

```{tip}
When setting a fresh Ubuntu system, you can choose to install docker right away. Then you can skip the Docker installation step below and direktly install docker desktop after installing Conda.
```

This tutorial walks you through preparing a fresh Ubuntu Leapton system with all necessary development dependencies:
- PyCharm (IDE)
- Conda (Python environment manager)
- Docker (for containerized workflows)
- A test environment using Conda

You can follow along with the accompanying video or execute the commands directly below.

```{raw} html
<video width="640" controls poster="https://raw.githubusercontent.com/Alexander-Nasuta/Alexander-Nasuta/main/readme_images/logo.png">
  <source src="https://rwth-aachen.sciebo.de/s/mxoxLmH34NBeN5Z/download/KIOptiPack-System-Setup-Tutorial.mov" type="video/mp4">
  Your browser does not support the video tag.
</video>
```

## 🧾 1. Check Your System Information

Before starting, verify your system configuration.

```bash
lsb_release -a
```

```{note}
You can also view system details under *Settings → System → System Details*
```

Check Current Python Installation (if any):
```bash
which python3
```
Check if Conda Is Installed
```bash
conda --version
```
Check if Docker Is Installed
```bash
docker --version
```


## 🐍 2. Install Conda (Anaconda Distribution)

[Anaconda](https://www.anaconda.com/) provides a powerful environment and package manager called **Conda**, making it easy to manage Python versions, dependencies, and isolated environments.  
We’ll install the full Anaconda distribution, which includes Conda, Python, and several commonly used scientific packages.

```{tip}
If you prefer a lighter setup, you can also install Miniconda instead of the full Anaconda distribution.
It provides Conda without the bundled packages and can be downloaded from [conda.io](https://docs.conda.io/en/latest/miniconda.html).
```

### Navigate to your downloads folder
```bash
cd ~/Downloads
```

### Download the installer
```bash
curl -O https://repo.anaconda.com/archive/Anaconda3-2025.06-0-Linux-x86_64.sh
```
This will download the latest Anaconda installer for 64-bit Linux systems.

### Verify installer integrity
Before running the installer, verify its checksum to ensure the file wasn’t corrupted or tampered with.
Replace `<INSTALLER-FILENAME>` with the exact name of the downloaded file.

```bash
shasum -a 256 ~/Anaconda3-2025.06-0-Linux-x86_64.sh
```

```{note}
Compare the printed checksum with the value listed on the [Anaconda archive page](https://repo.anaconda.com/archive/).
If they match, the installer is safe to run.
```
### Run the installer
```bash
bash ~/Anaconda3-2025.06-0-Linux-x86_64.sh
```
Follow the on-screen prompts to accept the license and choose an installation location (default is recommended).


### Refresh your terminal

To load Conda into your current shell session, either restart your terminal or run:

```bash
source ~/.bashrc
```

### Test your installation
```bash
conda --version
```

### Install Anaconda Navigator dependencies
The Anaconda Navigator GUI requires some additional system libraries on Ubuntu:
```bash
sudo apt-get install libgl1-mesa-glx libegl1-mesa libxrandr2 libxss1 \
libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6
```


## 🐳 3. Install Docker
```{important}
The following steps follow the official Docker installation guide for Ubuntu.
These might change over time, so always refer to the [official Docker documentation](https://docs.docker.com/engine/install/ubuntu/) for the most up-to-date instructions.
```

[Docker](https://www.docker.com/) is an essential tool for creating reproducible, isolated environments.  
It allows you to run your software, databases, or services inside containers that behave identically across systems — making development and deployment more reliable.

In this section, you’ll install and verify Docker on **Ubuntu Lapton**, following the recommended practices from the official Docker documentation.

---

## 🔍 Check for Existing Docker Packages

Before installing Docker, remove any unofficial or older packages that might conflict with the official version.

```bash
dpkg -l docker.io docker-doc docker-compose docker-compose-v2 podman-docker
```
If any of these packages are installed, uninstall them using:
```
sudo apt-get remove docker.io docker-doc docker-compose docker-compose-v2 podman-docker
```

Then, clean up residual dependencies:
```
sudo apt-get autoremove
```

### 🧩 Check Virtualization Support
Docker relies on virtualization features of your CPU to run efficiently.
To confirm that virtualization is supported and enabled:

```bash
lscpu | grep "Virtualization"
```

```{tip}
If the command shows no output, it means virtualization is not enabled.
You’ll need to enable it in your BIOS or UEFI settings (usually under “Advanced → CPU Configuration”).
```

### ⚙️ Verify KVM Support
KVM (Kernel-based Virtual Machine) enhances container performance.
Check if KVM modules are active:

```bash
lsmod | grep kvm
```
If you see no output, load the module manually:
```bash
sudo modprobe kvm
```

### 🔒 Check Ownership of /dev/kvm
Verify that your user has the correct permissions for virtualization:

```bash
ls -al /dev/kvm
```

If you don’t own the device or lack permission, add yourself to the kvm group:
```bash
sudo usermod -aG kvm $USER
```

```{note}
You’ll need to log out and log back in (or restart your session) for this change to take effect.
```

### 🧰 Ensure QEMU Version ≥ 5.2
Some Docker configurations rely on QEMU, especially when running virtual machines.
Update your system and install QEMU if necessary:

```bash
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install qemu-system
```

### 🪄 Verify Init System
Ubuntu uses systemd as its init system by default, which is compatible with Docker.
You can verify this with:
```
ps --no-headers -o comm 1
```
If the output is `systemd`, you’re good to go!

### 🖥️ Check Desktop Environment
Docker Desktop requires a supported graphical environment if you plan to use it (e.g. for managing containers visually).
Check your current desktop environment:
```bash
echo $XDG_CURRENT_DESKTOP
```
Supported environments include **GNOME**, **KDE**, or **MATE**.

### 🔐 (Optional) Install Password Manager
```{note}
This is not needed to execute the blueprints in this documentation.
```
If you plan to access private Docker registries, install a password manager like pass to securely store credentials.
```bash
sudo apt-get install pass
```

### 📦 Set Up Docker’s Official APT Repository

Now we’ll add Docker’s official repository to ensure you get the latest stable version.

1. Uninstall any old versions (if you skipped above):
```bash
sudo apt-get remove docker docker-engine docker.io containerd runc
```
2. Install required packages:
```bash
sudo apt-get update
sudo apt-get install ca-certificates curl gnupg lsb-release
```
3. Add Docker’s GPG key:
```bash
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
```
4. Add the stable Docker repository:
```
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
```
5. Update the package index:
```bash
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

### 🧪 Verify Installation
After installation, confirm that Docker is running correctly.
Check Docker version:
```bash
docker --version
```
Run the classic “hello world” test:
```bash
sudo docker run hello-world
```
if you see the “Hello from Docker!” message, the installation was successful.

### Install Docker Desktop
Docker Desktop provides a user-friendly interface for managing containers and images.
Follow the official guide to install Docker Desktop on Linux:
```bash
https://docs.docker.com/desktop/install/linux-install/
```



## 💻 4. Install PyCharm
PyCharm is a professional Python IDE developed by JetBrains, offering excellent debugging, environment management, and project tools.
Check your system architecture
```
uname -m
```
This confirms whether your machine is 64-bit (x86_64), required for PyCharm.
Visit the [JetBrains download page](https://www.jetbrains.com/pycharm/download/) and choose Community (free) or Professional:

## 5. Set Up a Conda Test Environment
To verify your setup, create a simple Conda environment and run Python inside it.
### Create and activate the environment
```
conda create -n test-env python=3.11
conda activate test-env
```
### Check Python version inside Conda
```
python --version
```
You should see Python 3.11.x, confirming that Conda is managing your Python environment correctly.
