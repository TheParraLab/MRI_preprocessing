#!/usr/bin/env python3
"""
Install Docker and NVIDIA Container Toolkit for Linux.

This script assumes Ubuntu/Debian-based systems. It is designed for
the MRI Preprocessing pipeline which runs inside Docker containers.

Usage:
    sudo python3 install.py
"""

import os
import subprocess
import sys


def run_cmd(command, **kwargs):
    """Run a command and raise on failure."""
    return subprocess.run(command, shell=True, check=True, **kwargs)


def check_gpu_presence():
    """Check if an NVIDIA or AMD GPU is present."""
    try:
        output = subprocess.check_output("lspci | grep -E 'NVIDIA|AMD'", shell=True).decode()
        return bool(output.strip())
    except subprocess.CalledProcessError:
        return False


def is_docker_installed():
    """Check if Docker is already installed."""
    try:
        subprocess.run(["docker", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def is_docker_gpu_configured():
    """Check if Docker is configured to use GPUs."""
    try:
        subprocess.run(
            ["docker", "run", "--rm", "--gpus", "all", "nvidia/cuda:11.5.2-base-ubuntu20.04", "nvidia-smi"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
        )
        print("Docker is configured to use the GPU.")
        return True
    except subprocess.CalledProcessError:
        return False


def install_docker():
    print("Installing Docker on Linux...")
    print("This process will follow the instructions from https://docs.docker.com/engine/install/ubuntu/")
    print("Post-installation steps are included, ensuring sudo is not required to run Docker commands.")
    print("Please make sure you have sudo privileges.")
    ans = input("Type 'yes' to continue: ")
    if ans.lower() != 'yes':
        print("Installation aborted.")
        sys.exit(0)

    run_cmd(["sudo", "apt-get", "update"])
    run_cmd(["sudo", "apt-get", "install", "-y", "ca-certificates", "curl"])
    run_cmd(["sudo", "install", "-m", "0755", "-d", "/etc/apt/keyrings"])
    run_cmd(['sudo', "curl", "-fsSL", "https://download.docker.com/linux/ubuntu/gpg", "-o", "/etc/apt/keyrings/docker.asc"])
    run_cmd(["sudo", "chmod", "a+r", "/etc/apt/keyrings/docker.asc"])

    arch = subprocess.check_output(["dpkg", "--print-architecture"]).decode().strip()
    version_codename = subprocess.check_output("source /etc/os-release && echo $VERSION_CODENAME", shell=True).decode().strip()

    run_cmd(['sudo', "sh", "-c", f'echo "deb [arch={arch} signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu {version_codename} stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null'])
    run_cmd(["sudo", "apt-get", "update"])
    run_cmd(["sudo", "apt-get", "install", "-y", "docker-ce", "docker-ce-cli", "containerd.io", "docker-buildx-plugin", "docker-compose-plugin"])

    if subprocess.run(["getent", "group", "docker"], capture_output=True).returncode != 0:
        run_cmd(["sudo", "groupadd", "docker"])
    run_cmd(["sudo", "usermod", "-aG", "docker", os.environ["USER"]])

    run_cmd(["sudo", "systemctl", "enable", "docker.service"])
    run_cmd(["sudo", "systemctl", "enable", "containerd.service"])
    run_cmd(["sudo", "systemctl", "start", "docker.service"])

    run_cmd(["sudo", "docker", "--version"])
    run_cmd(["sudo", "docker", "run", "hello-world"])

    print('#' * 50)
    print("Docker installation complete.")
    print("This project requires Docker to run as a non-root user.")
    print("Group changes have been generated, but not applied.")
    print("Please logout and login again to apply group changes.")
    print('#' * 50)


def install_container_toolkit():
    print("Installing NVIDIA Container Toolkit on Linux...")
    print("This process will follow the instructions from https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html")
    print("Please make sure you have sudo privileges.")
    ans = input("Type 'yes' to continue: ")
    if ans.lower() != 'yes':
        print("Installation aborted.")
        sys.exit(0)

    run_cmd("""
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \\
    && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \\
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \\
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    """, shell=True)

    run_cmd(["sudo", "apt-get", "update"])
    run_cmd(["sudo", "apt-get", "install", "-y", "nvidia-container-toolkit"])
    run_cmd("sudo nvidia-ctk runtime configure --runtime=docker", shell=True)
    run_cmd(["sudo", "systemctl", "restart", "docker"])


def main():
    if os.name != 'posix':
        print("Unsupported OS. This script requires Linux.")
        sys.exit(1)

    print(f"Detected OS: Linux")

    if not is_docker_installed():
        install_docker()
    else:
        print("Docker is already installed.")

    if not check_gpu_presence():
        print("No GPU detected. The pipeline requires a GPU for preprocessing.")
        sys.exit(1)
    else:
        print("GPU detected.")

    if not is_docker_gpu_configured():
        install_container_toolkit()

    print("\nInstallation complete.")


if __name__ == "__main__":
    main()
