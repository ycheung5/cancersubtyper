from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from database import init_db, initialize_models

# Import routers
from routers.auth import router as auth_router
from routers.users import router as users_router
from routers.projects import router as projects_router
from routers.jobs import router as jobs_router
from routers.visualization import router as visualization_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for database initialization."""
    print("Initializing database...")
    init_db()  # Create tables if they don't exist
    initialize_models()  # Ensure "BCtypeFinder" is added if missing
    print("Database initialized and models verified.")

    yield  # Allows FastAPI to continue running after setup
    print("Shutting down application...")


# Initialize FastAPI with lifespan
app = FastAPI(
    title="CancerSubtyper API",
    description="API service for CancerSubtyper Web-platform",
    version="1.0.0",
    lifespan=lifespan,  # Set the lifespan handler
)

# CORS Middleware (Allows frontend to communicate with backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(auth_router)
app.include_router(users_router)
app.include_router(projects_router)
app.include_router(jobs_router)
app.include_router(visualization_router)


# Root Endpoint (Health Check)
@app.get("/", tags=["Health Check"])
async def root():
    return {"message": "CancerSubtyper API is running!"}
