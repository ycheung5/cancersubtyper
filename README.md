# CancerSubtyper

## Abstract

**CancerSubtyper** is a web-based platform for deep learning-based cancer subtyping using DNA methylation data. It supports both supervised and semi-supervised workflows for predicting or discovering molecular subtypes. Users can upload methylation datasets (with or without subtype labels), run models, and explore interactive visualizations such as UMAP projections, CpG heatmaps, and Kaplan-Meier survival plots.

The platform currently includes:
- **BCtypeFinder** – a supervised classifier trained on TCGA-BRCA for intrinsic breast cancer subtype prediction.
- **CancerSubminer** – a semi-supervised model that performs subtype discovery or refinement with optional clustering constraints.

This tool is designed to be accessible to non-programmers while remaining robust enough for advanced molecular analysis.

---

## Table of Contents

- [Requirements](#requirements)
- [Architecture](#architecture)
- [Installing Docker and Docker Compose](#installing-docker-and-docker-compose)
- [Installation & Setup](#installation--setup)
- [Running the Application](#running-the-application)
- [Accessing Services](#accessing-services)
- [Development](#development)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Requirements

### Software Dependencies

Before you begin, you need to install the following software on your computer:

- **Docker** (version 20.10 or higher) - A platform that allows you to run applications in containers
- **Docker Compose** (version 2.0 or higher) - A tool for defining and running multi-container Docker applications
- **NVIDIA Docker Runtime** (for GPU support - optional but recommended) - Only needed if you have an NVIDIA GPU

### Hardware Recommendations

- **CPU**: 4 or more cores
- **RAM**: At least 16GB (32GB is recommended for best performance)
- **Disk**: At least 50GB of free storage space (more space for larger datasets)
- **GPU**: NVIDIA GPU with CUDA support (optional, speeds up model training/inference)

---

## Architecture

CancerSubtyper is a multi-service application consisting of:

- **Frontend (React + Vite)**: User interface for uploading data and viewing results
- **Backend API (FastAPI)**: REST API for managing users, projects, jobs, and models
- **PostgreSQL Database**: Stores user data, project metadata, and job information
- **Redis**: Message broker for asynchronous task queue
- **Celery Workers**: Background workers for running computationally intensive deep learning tasks
- **Flower**: Celery monitoring tool (optional)

---

## Installing Docker and Docker Compose

**⚠️ Important:** You must install Docker and Docker Compose before proceeding with the application setup.

### Step 1: Install Docker

Choose the instructions that match your operating system:

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

---

## Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/cancersubtyper.git
cd cancersubtyper
```

### 2. Set Up Environment Variables

You need to create configuration files (called `.env` files) for the application. These files contain important settings like passwords and database information.

**⚠️ Important for beginners:** 
- A `.env` file is a plain text file that stores configuration settings
- The `.env` filename starts with a dot, which may make it hidden on some systems
- You'll create these files using a text editor (like Notepad on Windows, TextEdit on Mac, or nano/vim on Linux)

#### Root `.env` (for PostgreSQL Database)

**Step 1:** Navigate to the project root directory (you should already be there after cloning):

```bash
pwd  # This shows your current directory - should show the cancersubtyper folder
```

**Step 2:** Create a new file called `.env` in the project root:

**On Linux or Mac:**
```bash
nano .env
```

**On Windows (using WSL):**
```bash
nano .env
```

**On Windows (using Notepad):**
- Open Notepad
- Click File → Save As
- Navigate to your cancersubtyper folder
- In the "File name" box, type: `.env`
- In the "Save as type" dropdown, select "All Files (*.*)"
- Click Save

**Step 3:** Add the following content to the `.env` file:

```env
# PostgreSQL Database Configuration
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_secure_password_here
POSTGRES_DB=cancersubtyper
```

**Step 4:** Replace `your_secure_password_here` with your own password. Example: `POSTGRES_PASSWORD=MySecurePassword123!`

**⚠️ Security Note**: 
- Use a strong password (at least 12 characters with letters, numbers, and symbols)
- Write this password down somewhere safe - you'll need it in the next step
- Examples of good passwords: `MySecurePass2024!` or `CancerSub#123`

**Step 5:** Save and close the file:
- **nano:** Press `Ctrl+X`, then `Y`, then Enter
- **Notepad:** Click File → Save (or press Ctrl+S)

#### API `.env` (Backend Configuration)

Now you need to create another `.env` file in the `api/` subdirectory (the folder that contains the backend code).

**Step 1:** Navigate to the `api/` directory:

```bash
cd api
```

**Step 2:** Create a new `.env` file:

**On Linux or Mac:**
```bash
nano .env
```

**On Windows (using WSL):**
```bash
nano .env
```

**On Windows (using Notepad):**
- Open Notepad
- Click File → Save As
- Navigate to the `api` folder inside your cancersubtyper folder
- In the "File name" box, type: `.env`
- In the "Save as type" dropdown, select "All Files (*.*)"
- Click Save

**Step 3:** Generate a JWT Secret Key first. This is a long random string used for security.

**On Linux or Mac:**
Open a NEW terminal window (keep the editor open). Run one of these commands:

```bash
# Using Python
python3 -c "import secrets; print(secrets.token_hex(64))"

# Using OpenSSL (alternative)
openssl rand -hex 64
```

**On Windows (WSL):**
Open a NEW terminal window and run:

```bash
python3 -c "import secrets; print(secrets.token_hex(64))"
```

**On Windows (PowerShell):**
```powershell
python -c "import secrets; print(secrets.token_hex(64))"
```

**Copy the entire output** (it will be a long string of letters and numbers like `a3f5b7c9...`). You'll paste this in the next step.

**Step 4:** Add this content to the `api/.env` file and customize:

```env
# Database URL - must match your PostgreSQL credentials from the root .env file
# Replace 'your_secure_password_here' with the SAME password you used in the root .env file
SQLALCHEMY_DATABASE_URL=postgresql://postgres:your_secure_password_here@db/cancersubtyper

# JWT Settings
JWT_SECRET_KEY=your_jwt_secret_key_here
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=30
JWT_REFRESH_TOKEN_EXPIRE_DAYS=7

# Storage
MAX_STORAGE_BYTES=21474836480  # 20GB in bytes
DATA_DIR=/app/data
FILE_WRITER_CHUNK_SIZE=1

# Celery & Redis
CELERY_BROKER_URL=redis://redis:6379/0
CELERY_RESULT_BACKEND=redis://redis:6379/0

# Path
CPG_INFO_FILE=/app/data/global/cpg_info.csv
NEMO_SCRIPT_FILE=/app/tasks/helper_scripts/run_nemo.R
SAMPLE_FILE=/app/data/global/sample
```

**Step 5:** Replace the placeholders in the `api/.env` file:
1. Replace `your_secure_password_here` in the SQLALCHEMY_DATABASE_URL line with the **exact same password** you used in the root `.env` file
2. Replace `your_jwt_secret_key_here` with the long random string you generated in Step 3

**Example of what your final file should look like:**

```env
SQLALCHEMY_DATABASE_URL=postgresql://postgres:postgres@db/cancersubtyper

# JWT Settings
JWT_SECRET_KEY=7f3e9a2b8c1d4e5f6a7b8c9d0e1f2a3b4c5d6e7asbv44321244d2e3f4a5b6c7d8e9f0a1
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=30
JWT_REFRESH_TOKEN_EXPIRE_DAYS=7

# Storage
MAX_STORAGE_BYTES=21474836480  # 20GB in bytes
DATA_DIR=/app/data
FILE_WRITER_CHUNK_SIZE=1

# Celery & Redis
CELERY_BROKER_URL=redis://redis:6379/0
CELERY_RESULT_BACKEND=redis://redis:6379/0

# Path
CPG_INFO_FILE=/app/data/global/cpg_info.csv
NEMO_SCRIPT_FILE=/app/tasks/helper_scripts/run_nemo.R
SAMPLE_FILE=/app/data/global/sample
```

**Step 6:** Save and close the file:
- **nano:** Press `Ctrl+X`, then `Y`, then Enter
- **Notepad:** Click File → Save (or press Ctrl+S)

**Step 7:** Go back to the project root directory:

```bash
cd ..
```

#### App `.env` (Frontend Configuration)

Finally, you need to create one more `.env` file in the `app/` subdirectory (the folder that contains the frontend code).

**Step 1:** Navigate to the `app/` directory:

```bash
cd app
```

**Step 2:** Create a new `.env` file:

**On Linux or Mac:**
```bash
nano .env
```

**On Windows (using WSL):**
```bash
nano .env
```

**On Windows (using Notepad):**
- Open Notepad
- Click File → Save As
- Navigate to the `app` folder inside your cancersubtyper folder
- In the "File name" box, type: `.env`
- In the "Save as type" dropdown, select "All Files (*.*)"
- Click Save

**Step 3:** Add the following content to the `app/.env` file:

```env
# API Configuration - this tells the frontend where to find the backend
VITE_API_BASE_URL=http://localhost:8000

# Polling intervals (in milliseconds) - how often to check for updates
VITE_POLL_PENDING=15000
VITE_POLL_PREPROCESSING=60000
VITE_POLL_RUNNING=60000
```

**⚠️ Important:** 
- For development on your local computer, use `http://localhost:8000` as shown above
- For production deployment, change `VITE_API_BASE_URL` to your actual API domain (e.g., `https://api.yourdomain.com`)
- You can leave the polling intervals as they are unless you have specific requirements

**Step 4:** Save and close the file:
- **nano:** Press `Ctrl+X`, then `Y`, then Enter
- **Notepad:** Click File → Save (or press Ctrl+S)

**Step 5:** Go back to the project root directory (this is important for the next step):

```bash
cd ..
```

---

### Verification: Check Your Setup

Before running the application, let's verify that all three `.env` files are created correctly.

**Step 1:** Check that all files exist:

**On Linux or Mac:**
```bash
ls -la .env api/.env app/.env
```

**On Windows (PowerShell):**
```powershell
Test-Path .env
Test-Path api\.env
Test-Path app\.env
```

All three files should exist. If any are missing, go back to the corresponding section above.

**Step 2:** You should now have:
- ✅ A `.env` file in the root directory (for PostgreSQL)
- ✅ A `.env` file in the `api/` directory (for the backend)
- ✅ A `.env` file in the `app/` directory (for the frontend)

**Important notes:**
- All three `.env` files must exist before you can run the application
- The PostgreSQL password in the root `.env` file must match the one in `api/.env`
- Make sure you're in the project root directory (type `pwd` to check)

---

## Running the Application

Now that all the configuration files are set up, you're ready to start the application!

**Important:** Make sure you're in the project root directory (the `cancersubtyper` folder). If you're not sure, run:
```bash
pwd
```
This should show something ending with `cancersubtyper`. If not, navigate there:
```bash
cd path/to/your/cancersubtyper
```

### Development Mode (with hot reload)

**Step 1:** Run the application with this command:

```bash
docker compose up --build
```

**What this command does:**
- `docker compose` - tells Docker to run multiple containers together
- `up` - starts the containers
- `--build` - rebuilds the images to include any code changes

**Step 2:** You'll see lots of output scrolling in your terminal. This is normal! Docker is:
1. Building all Docker images (this downloads necessary software like Python, Node.js, databases, etc.)
2. Starting PostgreSQL database
3. Starting Redis (message queue)
4. Starting API backend server
5. Starting Frontend web server
6. Starting Celery workers (for background tasks)
7. Enabling hot-reload (🧾 automatic updates when you change code)

**⚠️ First-time setup:** This step will take several minutes the first time (5-15 minutes) because Docker needs to download all the required software. Subsequent startups will be much faster (30-60 seconds).

**Step 3:** You'll know everything is ready when you see messages like:
- "Application startup complete"
- "Started server process"
- "Frontend development server running"

You can now access the application at http://localhost:5173

**Tip:** Keep this terminal window open. If you close it, the application will stop running.

**To stop the application:**
- Press `Ctrl+C` in the terminal
- Or in a new terminal window, run: `docker compose down`

### Production Mode

**⚠️ Production Mode is for deploying the application to a server, not for local development.**

If you're just getting started and testing the application on your own computer, you don't need this section. Continue with the next section "Accessing Services" instead.

If you need to deploy this to a production server, here are the basic steps:

1. **Build optimized frontend:**
   ```bash
   cd app
   npm run build
   cd ..
   ```

2. **Configure nginx service** in `compose.yml` (currently commented out)

3. **Run in detached mode** (runs in background):
   ```bash
   docker compose up -d
   ```

**Note:** Production deployment is an advanced topic. If you're a beginner, we recommend sticking with Development Mode for now.

### With GPU Support (Advanced)

**⚠️ This is optional and only needed if you have an NVIDIA GPU.**

By default, the application runs without GPU support. If you have an NVIDIA GPU and want to use it for faster processing:

#### Enable GPU Support:

1. **Install NVIDIA Container Toolkit:**
   - Linux: Follow instructions at https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
   - Mac/Windows: Install Docker Desktop with GPU support

2. **Verify GPU is accessible:**
   ```bash
   nvidia-smi  # Should show your GPU information
   ```

3. **GPU support is already configured** in `compose.yml` for the Celery worker and Flower services.

4. **Run the application:**
   ```bash
   docker compose up --build
   ```

#### Disable GPU Support (for systems without NVIDIA GPU):

If you don't have an NVIDIA GPU, you can disable GPU support to avoid errors:

1. **Edit `compose.yml`:**
   - Comment out the `deploy` sections for `celery_worker` and `flower` services (lines 64-72 and 90-98)
   - Or remove the GPU device reservations entirely

2. **Run the application:**
   ```bash
   docker compose up --build
   ```

**Note:** The application will work fine without GPU support, but model training and inference will be slower.

---

## Accessing Services

Once the application is running, you can access different parts of the system:

### 🌐 Main Application (What You'll Use Most)

**Web Application (Frontend):**
- **URL:** http://localhost:5173
- **Description:** This is the main user interface where you'll upload data, run jobs, and view results
- **How to access:** Open your web browser and go to `http://localhost:5173`
- **First time:** You'll need to create an account or sign in

### 🔧 Developer Tools (Optional)

These are mainly for developers, but you can explore them if curious:

**Backend API:**
- **URL:** http://localhost:8000
- **Description:** The backend server that handles requests
- **What it does:** Processes data, runs models, manages database

**API Documentation (Swagger UI):**
- **URL:** http://localhost:8000/docs
- **Description:** Interactive documentation showing all API endpoints
- **What you can do:** Try out API calls directly in your browser
- **For beginners:** You probably won't need this unless you want to integrate with the API programmatically

**Flower (Task Monitor):**
- **URL:** http://localhost:5556
- **Description:** Monitor background jobs (jobs that are processing your data)
- **What you can see:** Active jobs, completed jobs, and their status
- **When to use:** Check if your background jobs are running successfully

### 🗄️ Database Access (Advanced Only)

**PostgreSQL Database:**
- **Host:** localhost
- **Port:** 5432
- **When to use:** Only if you want to directly access the database
- **Tools needed:** Install a database client like pgAdmin or DBeaver
- **For beginners:** You don't need to access this directly - the web interface handles everything

### Summary

**To start using the application:**
1. Make sure Docker is running (see "Running the Application" section above)
2. Open your web browser
3. Go to http://localhost:5173
4. Create an account and start uploading data!

---

## Development

### Project Structure

```
cancersubtyper/
├── api/                    # FastAPI backend
│   ├── data/              # User data and global files
│   ├── helpers/           # Utility functions
│   ├── models.py          # SQLAlchemy database models
│   ├── repository/        # Database access layer
│   ├── routers/           # API route handlers
│   ├── schemas/           # Pydantic schemas
│   ├── tasks/             # Celery tasks and ML models
│   ├── main.py            # FastAPI app entry point
│   ├── config.py          # Configuration management
│   └── requirements.txt   # Python dependencies
│
├── app/                   # React frontend
│   ├── src/
│   │   ├── components/   # React components
│   │   ├── pages/        # Page components
│   │   ├── redux/        # State management
│   │   └── shared/       # Utilities and constants
│   ├── package.json      # Node.js dependencies
│   └── vite.config.js    # Vite configuration
│
├── compose.yml           # Docker Compose configuration
└── README.md            # This file
```

### Running Tests

```bash
# Backend tests
cd api
python -m pytest

# Frontend tests
cd app
npm test
```

### Making Changes

- **Backend**: Changes to Python files will trigger automatic reload (uvicorn --reload)
- **Frontend**: Changes to React files will hot-reload automatically (Vite HMR)
- **Database models**: If you modify `models.py`, you may need to create database migrations

---

## Troubleshooting

### Common Issues

**💡 Important:** If you run into problems, don't panic! These issues are common and usually have simple solutions.

#### 1. "Port Already in Use" Error

**What this means:** Another application is already using a port that CancerSubtyper needs (like 5432, 8000, or 5173).

**How to check what's using the port:**

**On Windows:**
```powershell
# Check port 5432 (PostgreSQL)
netstat -ano | findstr :5432

# Check port 8000 (API)
netstat -ano | findstr :8000

# Check port 5173 (Frontend)
netstat -ano | findstr :5173
```

**On Linux or Mac:**
```bash
# Check port 5432 (PostgreSQL)
lsof -i :5432

# Check port 8000 (API)
lsof -i :8000

# Check port 5173 (Frontend)
lsof -i :5173
```

**How to fix:**
1. **Option 1:** Stop the other application that's using the port
2. **Option 2:** Change the port in `compose.yml` (not recommended for beginners)

#### 2. "Database Connection Failed" Error

**What this means:** The API can't connect to the PostgreSQL database.

**Common causes:**
1. Docker containers aren't running
2. Password mismatch between the root `.env` and `api/.env` files
3. Database hasn't started yet (wait a few seconds)

**How to diagnose:**
```bash
# Check if containers are running
docker compose ps

# You should see: db, redis, api, app, celery_worker, flower all "Up"
```

**How to fix:**
1. **Check the `.env` files match:**
   - The password in the root `.env` file must match the password in `api/.env`
   - Copy them exactly - they need to be identical

2. **Restart the containers:**
   ```bash
   docker compose down
   docker compose up --build
   ```

#### 3. "Cannot connect to Docker" or "Cannot connect to the Docker daemon"

**What this means:** Docker isn't running on your computer.

**How to fix:**

**On Linux:**
```bash
# Start Docker
sudo systemctl start docker

# Enable Docker to start on boot (optional but helpful)
sudo systemctl enable docker
```

**On macOS or Windows:**
- Open Docker Desktop from your Applications folder
- Wait for the Docker whale icon to appear in your menu bar/system tray
- If Docker Desktop shows an error, try restarting it

#### 4. "Permission Denied" Error (Linux)

**What this means:** Your user doesn't have permission to run Docker.

**How to fix:**
```bash
# Add your user to the docker group
sudo usermod -aG docker $USER

# Log out and log back in (or run this command)
newgrp docker

# Try again
docker --version
```

#### 5. "File Not Found" for `.env` Files

**What this means:** The `.env` files weren't created correctly.

**How to fix:**
1. **Verify files exist:**
   ```bash
   ls -la .env api/.env app/.env  # Linux/Mac
   dir .env api\.env app\.env     # Windows
   ```

2. **If files are missing:** Go back to the "Set Up Environment Variables" section and create them again

3. **Check you're in the right directory:**
   ```bash
   pwd  # Should end with "cancersubtyper"
   ```

#### 6. Application Won't Start / Containers Keep Crashing

**How to diagnose:**
```bash
# Check what's failing
docker compose logs

# Check specific service logs
docker compose logs api        # Backend logs
docker compose logs app        # Frontend logs
docker compose logs celery_worker  # Worker logs
```

**Common causes:**
- Missing `.env` files
- Incorrect passwords in `.env` files
- Not enough disk space
- Not enough memory

#### 7. Frontend Shows "Cannot Connect to API"

**What this means:** The frontend can't talk to the backend.

**How to fix:**
1. **Check `app/.env` file:**
   ```env
   VITE_API_BASE_URL=http://localhost:8000
   ```

2. **Verify API is running:**
   ```bash
   curl http://localhost:8000/docs
   # Should show HTML output
   ```

3. **Restart the frontend:**
   ```bash
   docker compose restart app
   ```

### Viewing Logs

```bash
# All services
docker compose logs -f

# Specific service
docker compose logs -f api
docker compose logs -f celery_worker
docker compose logs -f app

# Last 100 lines
docker compose logs --tail=100 api
```

### Resetting the Database

**⚠️ Warning: This will delete ALL data including user accounts, projects, and job results!**

**When to use this:**
- When you want to start fresh and remove all data
- If the database becomes corrupted
- During development when you want to reset everything

**How to reset:**
```bash
# Stop all containers and remove volumes (this deletes the data)
docker compose down -v

# Start everything fresh
docker compose up --build
```

**Note:** After running this, you'll need to create a new account when you access the web application.

---

## Storage Management

User data is stored in `api/data/` directory with the following structure:

```
api/data/
├── global/              # Global reference data (CpG info, sample data)
└── user_{id}/          # Per-user directories
    └── project_{id}/   # Per-project directories
        ├── source/     # Source dataset
        ├── target/     # Target dataset
        ├── metadata/   # Clinical metadata
        └── job_{id}/   # Job-specific results
```

Default storage limit: **20GB per user** (configurable via `MAX_STORAGE_BYTES`)

---

## Security Best Practices

1. **Never commit `.env` files** - They contain sensitive credentials
2. **Use strong JWT secrets** - Generate with cryptographic tools
3. **Change default passwords** - Especially for PostgreSQL
4. **Use HTTPS in production** - Uncomment and configure nginx service
5. **Regularly update dependencies** - Check for security vulnerabilities
6. **Limit storage quotas** - Prevent disk exhaustion attacks
7. **Implement rate limiting** - Protect against brute force attacks

---

## Acknowledgments

This work was supported by:
- U.S. National Science Foundation (NSF) Awards #2004751, #2125798, #2344169, and #2319522
- National Institutes of Health (NIH) Grant #1R01AI179686-01A1
