import os
import subprocess
import sys
import ctypes
## Install.py #################################################################
# This script installs Docker and NVIDIA Container Toolkit on Linux.
# It also checks for the presence of a GPU and configures Docker to use the GPU.
###############################################################################
#TODO CHECKS:
# Windows setup pending implementation, unnecessary?

def run_as_admin(command):
    """Run a command in an elevated Command Prompt window and wait for it to complete."""
    # PowerShell command to run the specified command in a new elevated window
    ps_command = f'Start-Process cmd.exe -ArgumentList "/K, {command}" -Verb RunAs -Wait'
    try:
        # Execute the PowerShell command and wait for it to complete
        subprocess.run(["powershell", "-Command", ps_command], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to run command as admin: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

def check_gpu_presence(OS):
    try:
        if OS == "Linux":
            lspci_output = subprocess.check_output("lspci | grep -E 'NVIDIA|AMD'", shell=True).decode()
            return bool(lspci_output.strip())
        elif OS == "Windows":
            wmic_output = subprocess.check_output("wmic path win32_videocontroller get name", shell=True).decode()
            return bool(wmic_output.strip())
        else:
            print("Unsupported OS")
            return False
    except subprocess.CalledProcessError:
        # If the command fails, assume no GPU is present
        return False

def is_choco_installed():
    try:
        subprocess.run(["choco", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except subprocess.CalledProcessError:
        # Chocolatey is installed but there might be a problem with it
        return False
    except FileNotFoundError:
        # Chocolatey is not installed
        return False
    
def is_docker_installed():
    try:
        subprocess.run(["docker", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except subprocess.CalledProcessError:
        return False
    except FileNotFoundError:
        return False

def is_docker_gpu_configured(OS):
    if OS == 'Windows':
        try:
            subprocess.run(["docker", "run", "--rm", "--gpus", "all", "nvidia/cuda:11.0-base", "nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            print("Docker is configured to use the GPU.")
            return True
        except subprocess.CalledProcessError:
            print("Docker is not configured to use the GPU.")
            return False
    elif OS == 'Linux':
        try:
            subprocess.run(["docker", "run", "--rm", "--gpus", "all", "nvidia/cuda:11.5.2-base-ubuntu20.04", "nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            print("Docker is configured to use the GPU.")
            return True
        except subprocess.CalledProcessError:
            print("Docker is not configured to use the GPU.")
            return False

def install_docker(OS):
    if OS == 'Linux':
        print("Installing Docker on Linux...")
        print("Do you want to install Docker?")
        print("This process will follow the instructions from https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository")
        print("Post-installation steps are included, ensuring sudo is not required to run Docker commands.")
        print("Please make sure you have sudo privileges.")
        ANS = input("Type 'yes' to continue: ")
        if ANS.lower() == 'yes':
            # Install prerequisites
            subprocess.run(["sudo", "apt-get", "update"], check=True)
            subprocess.run(["sudo", "apt-get", "install", "ca-certificates", "curl"], check=True)
            subprocess.run(["sudo", "install", "-m", "0755", "-d", "/etc/apt/keyrings"], check=True)
            subprocess.run(["sudo", "curl", "-fsSL", "https://download.docker.com/linux/ubuntu/gpg", "-o", "/etc/apt/keyrings/docker.asc"], check=True)
            subprocess.run(["sudo", "chmod", "a+r", "/etc/apt/keyrings/docker.asc"], check=True)
    
            # Get the architecture and version codename from environment variables
            arch_result = subprocess.run(["dpkg", "--print-architecture"], capture_output=True, text=True)
            arch = arch_result.stdout.strip()
            version_codename_command = "source /etc/os-release && echo $VERSION_CODENAME"
            version_codename_result = subprocess.run(["bash", "-c", version_codename_command], capture_output=True, text=True)
            version_codename = version_codename_result.stdout.strip()

            # Add Docker repository
            subprocess.run(["sudo", "sh", "-c", f'echo "deb [arch={arch} signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu {version_codename} stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null'], check=True)
            subprocess.run(["sudo", "apt-get", "update"], check=True)

            # Install Docker
            subprocess.run(["sudo", "apt-get", "install", "docker-ce", "docker-ce-cli", "containerd.io", "docker-buildx-plugin", "docker-compose-plugin"], check=True)
        
            # Post-installation steps
            # Check if the group 'docker' exists, if not, create it
            try:
                subprocess.run(["getent", "group", "docker"], check=True)
                print("Group 'docker' already exists. Skipping group creation.")
            except subprocess.CalledProcessError:
                subprocess.run(["sudo", "groupadd", "docker"], check=True)
            # Add the current user to the 'docker' group
            subprocess.run(["sudo", "usermod", "-aG", "docker", os.environ["USER"]], check=True)
            
            # Apply group changes without logging out
            # causes process blocking, unable to properly implement with current setup
            #subprocess.run(['sudo', 'newgrp', 'docker'], check=True)

            # Enable Docker service
            subprocess.run(["sudo", "systemctl", "enable", "docker.service"], check=True)
            subprocess.run(["sudo", "systemctl", "enable", "containerd.service"], check=True)
            # Start Docker service
            subprocess.run(["sudo", "systemctl", "start", "docker.service"], check=True)

            # Testing Docker
            subprocess.run(["sudo", "docker", "--version"], check=True)
            subprocess.run(["sudo", "docker", "run", "hello-world"], check=True)
            print('#'*50)
            print('Docker installation complete.')
            print('This project will require docker to run as non-root user.')
            print('Group changes have been generated, but not applied.')
            print('Please logout and login again to apply group changes.')
            print('#'*50)
        else:
            print("Installation aborted.")
            sys.exit(0)

    elif OS == 'Windows':
        print('#'*50)
        print("Installing Docker on Windows...")
        print('This installation script relies on the Chocolatey package manager.')
        print('Do you want to install Docker?')
        print('WARNING: If not already running as admin, the script will prompt for admin privileges.')
        print('You will need to close the secondary window once installation completes to continue.')
        if input('Type "yes" to continue: ').lower() == 'yes':
            run_as_admin(command="choco install docker-desktop")
            print("Installation complete.")
            print('#'*50)

        else:
            print("Installation aborted.")
            sys.exit(0)
        # Assuming Chocolatey is already installed

def install_container_toolkit(OS):
    if OS == 'Linux':
        print("Installing NVIDIA Container Toolkit on Linux...")
        print("Do you want to install NVIDIA Container Toolkit?")
        print("This process will follow the instructions from https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installing-with-apt")
        print("Please make sure you have sudo privileges.")
        ANS = input("Type 'yes' to continue: ")
        if ANS.lower() == 'yes':
            # Add the package repositories
            subprocess.run("""
            curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
            && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
            """, shell=True, check=True)
            subprocess.run(["sudo", "apt-get", "update"], check=True)
            # Install the NVIDIA Container Toolkit
            subprocess.run(["sudo", "apt-get", "install", "-y", "nvidia-container-toolkit"], check=True)

            # Configure Docker to use the NVIDIA runtime
            subprocess.run("""
            sudo nvidia-ctk runtime configure --runtime=docker""", shell=True, check=True)
            subprocess.run(["sudo", "systemctl", "restart", "docker"], check=True)
        else:
            print("Installation aborted.")
            sys.exit(0)

    elif OS == 'Windows':
        print("Installing NVIDIA Container Toolkit on Windows...")
        # Assuming Chocolatey is already installed

def install(OS):
    if OS == 'Windows':
        if not is_choco_installed():
            print("Chocolatey is not installed on Windows.")
            print("Please install Chocolatey from https://chocolatey.org/install")
            sys.exit(1)
        else:
            print("Chocolatey is already installed on Windows")
    if not is_docker_installed():
        install_docker(OS)
    else:
        print("Docker is already installed on Linux.")

    GPU = check_gpu_presence(OS)
    if GPU:
        print("GPU detected.")
    else:
        print("No GPU detected.")
        sys.exit(1)
    if not is_docker_gpu_configured(OS):
        install_container_toolkit(OS)

def main():
    if os.name == 'nt':
        OS = 'Windows'
    elif os.name == 'posix':
        OS = 'Linux'
    else:
        print("Unsupported OS")
        sys.exit(1)
    print(f"Detected OS: {OS}")
    install(OS)

    print("Installation complete.")

if __name__ == "__main__":
    main()