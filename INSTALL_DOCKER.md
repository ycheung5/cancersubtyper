## Installing Docker and Docker Compose

**⚠️ Important:** You must install Docker and Docker Compose before proceeding the CancerSubtyper installation.

### Step 1: Install Docker

Choose the instructions that match your operating system:

#### **For Windows:**

1. **Install WSL2 (Windows Subsystem for Linux):**
   - Open PowerShell as Administrator (right-click Start menu → Windows PowerShell (Admin))
   - Copy and paste this command, then press Enter:
     ```powershell
     wsl --install
     ```
   - Restart your computer when prompted

2. **After restart, install Docker Desktop:**
   - Go to: https://www.docker.com/products/docker-desktop/
   - Click "Download for Windows"
   - Open the downloaded file and follow the installation wizard
   - Check the box for "Use WSL 2 instead of Hyper-V" during installation

3. **Verify installation:**
   - Open a WSL terminal or PowerShell
   - Type: `docker --version`
   - Press Enter
   - You should see a version number

#### **For macOS:**

1. **Download Docker Desktop for Mac:**
   - Go to: https://www.docker.com/products/docker-desktop/
   - Click the big blue button that says "Download for Mac"
   - Choose the version for your Mac chip:
     - If you have an Apple Silicon Mac (M1, M2, M3), choose "Mac with Apple chip"
     - If you have an Intel Mac, choose "Mac with Intel chip"

2. **Open the downloaded file** (it will be in your Downloads folder)

3. **Drag the Docker icon** into your Applications folder

4. **Open Docker Desktop** from your Applications folder

5. **Follow the setup wizard:**
   - Click "Open" when asked about security
   - You may be asked to enter your Mac password to grant permissions
   - Click "Finish" when setup completes

6. **Verify installation:**
   - Open a Terminal (press Command + Space, type "Terminal", press Enter)
   - Type: `docker --version`
   - Press Enter
   - You should see a version number like: `Docker version 24.0.0`
     
#### **For Linux (Ubuntu/Debian):**

We recommend using the official Docker installation script, which is the easiest method:

1. **Open a terminal** (if you're on Windows, use WSL2 or a Linux virtual machine)

2. **Run the following command to download and install Docker:**
   ```bash
   curl -fsSL https://get.docker.com -o get-docker.sh
   ```

   Press its Enter/Return key on your keyboard. You should see the script download.

3. **Run the installation script:**
   ```bash
   sudo sh get-docker.sh
   ```

   You'll be asked to enter your password. Type your password and press Enter. Note: The cursor won't move while typing passwords - this is normal for security.

4. **Add your user to the docker group** (this allows you to run Docker without typing 'sudo' every time):
   ```bash
   sudo usermod -aG docker $USER
   ```

5. **Log out and log back in** for the group change to take effect. Or run this command to apply it immediately:
   ```bash
   newgrp docker
   ```

6. **Verify Docker is installed correctly:**
   ```bash
   docker --version
   ```

   You should see something like: `Docker version 24.0.0, build 371ceee`

7. **Test Docker by running a simple container:**
   ```bash
   docker run hello-world
   ```

   If successful, you'll see a message saying "Hello from Docker!"


### Step 2: Install Docker Compose

Docker Compose is usually included with Docker Desktop for Mac and Windows. For Linux, you may need to install it separately.

#### **For Linux (automatic install):**

Run these commands one by one in your terminal:

```bash
# Download the latest Docker Compose release
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose

# Make the file executable
sudo chmod +x /usr/local/bin/docker-compose

# Create a symbolic link
sudo ln -s /usr/local/bin/docker-compose /usr/bin/docker-compose
```

#### **Verify Docker Compose installation:**

```bash
docker-compose --version
```

You should see something like: `Docker Compose version v2.21.0`

### Step 3: Start Docker (if needed)

- **Linux:** Docker usually starts automatically. If you get an error saying "Cannot connect to Docker", run:
  ```bash
  sudo systemctl start docker
  sudo systemctl enable docker  # This makes Docker start automatically on boot
  ```

- **macOS/Windows:** Make sure Docker Desktop is running (you should see a Docker whale icon in your menu bar or system tray)

### Step 4: Verify Both Are Working

Run these two commands in your terminal to make sure everything is ready:

```bash
docker --version
docker-compose --version
```

Both commands should return version numbers. If you see any errors, refer to the troubleshooting section at the end of this README.
