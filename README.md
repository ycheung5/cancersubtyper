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
- [Installation & Setup](#installation--setup)
- [Environment Configuration](#environment-configuration)
- [Running the Application](#running-the-application)
- [Accessing Services](#accessing-services)
- [Development](#development)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Requirements

### Software Dependencies
- **Docker** (version 20.10 or higher)
- **Docker Compose** (version 2.0 or higher)
- **NVIDIA Docker Runtime** (for GPU support - optional but recommended)

### Hardware Recommendations
- **CPU**: 4+ cores
- **RAM**: 16GB minimum, 32GB recommended
- **Disk**: 50GB free space (more for large datasets)
- **GPU**: NVIDIA GPU with CUDA support (optional, for faster model training/inference)

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

## Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/cancersubtyper.git
cd cancersubtyper
```

### 2. Set Up Environment Variables

You'll need to create `.env` files for different parts of the application. We provide `.env.example` templates to help you get started.

#### Root `.env` (for PostgreSQL)

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Edit `.env` and configure the PostgreSQL credentials:

```env
# PostgreSQL Database Configuration
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_secure_password_here
POSTGRES_DB=cancersubtyper
```

**⚠️ Security Note**: Change the default password to a strong, unique password!

#### API `.env` (Backend Configuration)

Create an `.env` file in the `api/` directory:

```bash
cp api/.env.example api/.env
```

Edit `api/.env` with your configuration. **Important fields to customize:**

```env
# Database URL - must match your PostgreSQL credentials
SQLALCHEMY_DATABASE_URL=postgresql://postgres:your_secure_password_here@db/cancersubtyper

# JWT Secret - MUST be changed to a secure random string
JWT_SECRET_KEY=your_jwt_secret_key_here

# Storage limits
MAX_STORAGE_BYTES=21474836480  # 20GB in bytes
```

**🔑 Generating a JWT Secret:**

Use one of these commands to generate a secure JWT secret:

```bash
# Using Python
python -c "import secrets; print(secrets.token_hex(64))"

# Using OpenSSL
openssl rand -hex 64

# Using Node.js
node -e "console.log(require('crypto').randomBytes(64).toString('hex'))"
```

#### App `.env` (Frontend Configuration)

Create an `.env` file in the `app/` directory:

```bash
cp app/.env.example app/.env
```

Edit `app/.env` with the API base URL:

```env
# API Configuration
VITE_API_BASE_URL=http://localhost:8000

# Polling intervals (in milliseconds)
VITE_POLL_PENDING=15000
VITE_POLL_PREPROCESSING=60000
VITE_POLL_RUNNING=60000
```

For production, change `VITE_API_BASE_URL` to your actual API domain.

---

## Running the Application

### Development Mode (with hot reload)

```bash
docker compose up --build
```

This will:
1. Build all Docker images
2. Start PostgreSQL, Redis, API, Frontend, and Celery workers
3. Enable hot-reload for both frontend and backend code changes

### Production Mode

For production deployment, you may want to:

1. Build optimized frontend:
   ```bash
   cd app
   npm run build
   ```

2. Uncomment the nginx service in `compose.yml`

3. Run without the `--build` flag after initial setup:
   ```bash
   docker compose up -d
   ```

### With GPU Support

If you have an NVIDIA GPU, ensure you have the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed, then run:

```bash
docker compose up --build
```

The Celery worker and Flower services are already configured for GPU access in the `compose.yml`.

---

## Accessing Services

Once the application is running, you can access:

| Service | URL | Description |
|---------|-----|-------------|
| **Frontend** | http://localhost:5173 | React web application |
| **Backend API** | http://localhost:8000 | FastAPI REST API |
| **API Documentation** | http://localhost:8000/docs | Interactive Swagger UI |
| **Flower (Celery Monitor)** | http://localhost:5556 | Task queue monitoring |
| **PostgreSQL** | localhost:5432 | Database (use a client like pgAdmin) |

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

#### 1. Port Already in Use

If you get an error like "port is already allocated":

```bash
# Check what's using the port
netstat -ano | findstr :5432  # Windows
lsof -i :5432                 # Linux/Mac

# Stop the conflicting service or change the port in compose.yml
```

#### 2. Database Connection Failed

- Ensure PostgreSQL is running: `docker compose ps`
- Check credentials in `.env` match `api/.env`
- Verify the database URL format: `postgresql://user:password@db/database`

#### 3. Celery Worker Not Processing Tasks

- Check Redis is running: `docker compose ps redis`
- View worker logs: `docker compose logs celery_worker`
- Ensure the broker URL is correct in `api/.env`

#### 4. GPU Not Detected

- Verify NVIDIA drivers: `nvidia-smi`
- Check Docker has GPU access: `docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi`
- Review the Celery worker deployment section in `compose.yml`

#### 5. Frontend Can't Connect to API

- Ensure `VITE_API_BASE_URL` in `app/.env` is correct
- Check CORS settings in the backend
- Verify the API is running: `curl http://localhost:8000/docs`

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

**⚠️ Warning: This will delete all data!**

```bash
docker compose down -v
docker compose up --build
```

---

## Storage Management

User data is stored in `api/data/` directory with the following structure:

```
api/data/
├── global/              # Global reference data (CpG info, sample data)
└── user_{id}/          # Per-user directories
    └── project_{id}/   # Per-project directories
        ├── source/     # Source dataset
        ├── target/     # Target dataset (optional)
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

## Citation

If you use CancerSubtyper in your research, please cite:

```bibtex
@article{bctypefinder2025,
  title={BCtypeFinder: Deep Learning-Based Breast Cancer Subtype Prediction},
  journal={Genetic Testing and Molecular Biomarkers},
  year={2025},
  url={https://www.liebertpub.com/doi/abs/10.1177/15578666251380233}
}
```

---

## Acknowledgments

This work was supported by:
- U.S. National Science Foundation (NSF) Awards #2004751, #2125798, #2344169, and #2319522
- National Institutes of Health (NIH) Grant #1R01AI179686-01A1

---

## License

[Add your license information here]

---

## Contact

For questions, issues, or contributions, please [open an issue](https://github.com/your-username/cancersubtyper/issues) on GitHub.
