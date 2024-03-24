# Running TMU development in development containers

The recommended way of doing development using TMU is through devcontainers. This is because you get a unified development environment that is independent of your underlying system, making it much more trivial to debug when issues appear, and also is much faster to get up and running.
This guide shows how to run TMU with VSCODE both in remote development with SSH with devcontainers, and finally with Remote SSH combined with Devcontainers

# 0. Prerequisites

- Docker installed on your local machine. **Optional - When not using remote SSH**
- Visual Studio Code installed.
- Remote - Containers extension installed in VSCode.
- Ensure Git is installed on your system. This guide utilizes Git Bash for command execution across all operating systems, including Windows.

# 1. SSH Configuration

### Step 1: Generate SSH Keys

Regardless of your operating system, the first step is to generate an SSH key pair if you haven't done so already or wish to create a new pair for this connection. Open Git Bash and type:

```bash
ssh-keygen -t rsa -b 4096 -C "your_email@example.com"

```

Replace `your_email@example.com` with your actual email address for identification purposes. When prompted to "Enter a file in which to save the key," press Enter to accept the default location. You will then be asked to enter a passphrase; you can choose to enter one for added security or press Enter to proceed without a passphrase.

### Step 2: Copy the SSH Key to Your Remote Machine Using `ssh-copy-id`

The `ssh-copy-id` script offers a convenient way to install your public key in a remote machine's `~/.ssh/authorized_keys` file, enabling password-less SSH access. This tool is available on Linux, macOS, and through Git Bash on Windows. Here's how to use it:

### For Linux, macOS, and Windows (Using Git Bash)

1. Open your terminal or Git Bash (for Windows users).
2. Execute the following command, replacing `<username>` with your user on the remote machine and `cair-gpuXX.uia.no` with the remote machine's hostname or IP address:

```bash
ssh-copy-id <username>@cair-gpuXX.uia.no

```

1. You will be prompted to enter the remote machine's password. After successfully authenticating, your SSH key will be added to the remote machine's `~/.ssh/authorized_keys` file.

### Step 3: Verify Your SSH Setup

To test your new SSH key setup, attempt to SSH into your remote machine:

```bash
ssh <username>@cair-gpuXX.uia.no

```

If the setup is correct, you should gain access without being prompted for the remote user's password.

### For Multiple Remote Machines

If you're planning to establish password-less SSH connections with multiple remote machines, simply repeat the above process for each, ensuring your public SSH key is added to each machine's `~/.ssh/authorized_keys` file.

# 2. Development using a Devcontainer

# Step 1: Create the Docker-Compose File

Create a **`docker-compose.yml`** file within your **`.devcontainer`** directory. This file will define your service, including the use of GPUs.

### CUDA Enabled Configuration

**docker-compose.yml** (**`./.devcontainer/docker-compose.yml`**):

```yaml
version: '3.8'
services:
  tmu-development:
    build:
      context: .
      dockerfile: Dockerfile
    volumes:
      - ..:/app
    deploy:
      resources:
        reservations:
          devices:
          - driver: nvidia
            capabilities: [gpu]
            count: 1 # Assign number of GPUs or use 'all' to assign all available GPUs
```

### CPU-only Enabled Configuration

**docker-compose.yml** (**`./.devcontainer/docker-compose.yml`**):

```yaml
version: '3.8'
services:
  tmu-development:
    build:
      context: .
      dockerfile: Dockerfile
    volumes:
      - ..:/app

```

## Step 2: Create a Dockerfile

### CUDA Enabled Configuration

**Dockerfile** (**`./.devcontainer/Dockerfile`**):

```docker
FROM nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04

# Install Python and other dependencies
RUN apt-get update && apt-get install -y python3-pip

# Install TMU?, other relevant stuff

WORKDIR /app
COPY . /app

# You should have a requirements.txt to define your dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

CMD [ "tail", "-f", "/dev/null" ]
```

### CPU-only Enabled Configuration

**Dockerfile** (**`./.devcontainer/Dockerfile`**):

```docker
FROM ubuntu:22.04

# Install Python and pip
RUN apt-get update && apt-get install -y python3 python3-pip

# Optionally install other dependencies, tools, etc.

WORKDIR /app
COPY . /app

# If you have a requirements.txt, install dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

CMD [ "tail", "-f", "/dev/null" ]

```

## Step 3: Create devcontainer.json for Docker-Compose

Create **`devcontainer.json`**

**devcontainer.json** (**`./.devcontainer/devcontainer.json`**):

```json

{
    "name": "TMU Devcontainer",
    "dockerComposeFile": "docker-compose.yml",
    "service": "tmu-development",
    "workspaceFolder": "/app",
    "extensions": [
        "ms-python.python",
    ],
    "forwardPorts": [],
    "postCreateCommand": "echo 'Devcontainer is ready'",
    "remoteUser": "root"
}
```

## **Using the Setup**

- After configuring your **`.devcontainer`** directory with the **`Dockerfile`**, **`docker-compose.yml`**, and **`devcontainer.json`**, open your project in VSCode.
- VSCode may prompt you to reopen the project in a container. If not, you can manually do so by opening the Command Palette (**`F1`** or **`Ctrl+Shift+P`**/**`Cmd+Shift+P`**) and selecting "Remote-Containers: Reopen in Container".
- This will build your Docker container as defined, including the necessary GPU assignments for CUDA development.

# **3. Development Using Devcontainers on a Remote Machine (SSH)**

Running your development environment on a remote machine can provide significant performance benefits, especially for resource-intensive tasks. This setup requires a bit more initial configuration but has substantial advantages of powerful remote resources and a consistent development environment, like the DGX-2 machines.

## **Prerequisites**

- The remote machine must have Docker installed and running. (This is typically already done unless its your own remote machine)
- SSH access to the remote machine is set up (refer to the SSH setup guide provided earlier).
- VSCode and the Remote Development extension pack are installed on your local machine.

## **Setting Up Your Remote Devcontainer Environment**

1. **Connect to Your Remote Machine via SSH**: Open VSCode, then open the Command Palette and select "Remote-SSH: Connect to Host...". Choose your remote machine from the list or add a new SSH connection.
2. **Initialize Your Project on the Remote Machine**: You can clone your repository or access your project files on the remote machine. This might involve using Git commands within the terminal in VSCode once connected to the remote machine.
3. **Configure the Devcontainer**: Similar to the local setup, create a **`.devcontainer`** directory in your project on the remote machine with a **`Dockerfile`** and **`devcontainer.json`**. These files might already exist if you cloned a repository already configured for Devcontainer development.
4. **Open Your Project in a Container Over SSH**: With the remote SSH connection active and your project open in VSCode, use the Command Palette to select "Remote-Containers: Reopen in Container". This will build and start the container on the remote machine, with VSCode connecting to it over SSH.
5. **Start Developing Remotely**: You can now develop directly on the remote machine, utilizing its resources while benefiting from a consistent, containerized environment controlled by your Devcontainer configuration.