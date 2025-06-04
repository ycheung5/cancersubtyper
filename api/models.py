from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, Enum, Text, BigInteger
from sqlalchemy.orm import relationship
from database import Base
import enum

from helpers.timezone import get_utc_time


# ------------------- User Roles -------------------
class UserRole(str, enum.Enum):  # Store Enum as a string
    ADMIN = "admin"
    USER = "user"

# ------------------- Users Table -------------------
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, nullable=False)
    email = Column(String, unique=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    role = Column(Enum(UserRole), default=UserRole.USER, nullable=False)
    timezone = Column(String, default="America/New_York")
    storage_used = Column(BigInteger, default=0)

    projects = relationship("Project", back_populates="owner")
    jobs = relationship("Job", back_populates="user")  # Fix missing relationship

# ------------------- Projects Table -------------------
class Project(Base):
    __tablename__ = "projects"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    tumor_type = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    target_file = Column(String, nullable=True)
    source_file = Column(String, nullable=True)
    metadata_file = Column(String, nullable=True)
    created_at = Column(DateTime, default=get_utc_time)
    edited_at = Column(DateTime, default=get_utc_time, onupdate=get_utc_time)

    owner = relationship("User", back_populates="projects")
    jobs = relationship("Job", back_populates="project", cascade="all, delete-orphan")

# ------------------- Models Table (AI Models) -------------------
class Model(Base):
    __tablename__ = "models"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False)
    version = Column(String, nullable=False, default="1.0")  # Model version
    description = Column(Text, nullable=True)  # Use Text for descriptions

    jobs = relationship("Job", back_populates="model")

# ------------------- Job Status Enum -------------------
class JobStatusEnum(str, enum.Enum):  # Store Enum as a string
    PENDING = "Pending"
    PREPROCESSING = "Preprocessing"
    RUNNING = "Running"
    COMPLETED = "Completed"
    FAILED = "Failed"

# ------------------- Jobs Table -------------------
class Job(Base):
    __tablename__ = "jobs"

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    model_id = Column(Integer, ForeignKey("models.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    status = Column(Enum(JobStatusEnum), default=JobStatusEnum.PENDING, nullable=False)
    created_at = Column(DateTime, default=get_utc_time)
    started_at = Column(DateTime, nullable=True)
    finished_at = Column(DateTime, nullable=True)

    project = relationship("Project", back_populates="jobs")
    model = relationship("Model", back_populates="jobs")
    user = relationship("User", back_populates="jobs")
