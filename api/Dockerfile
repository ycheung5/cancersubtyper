# syntax=docker/dockerfile:1.4
ARG INSTALL_R=false
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install base system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    wget \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Add conditional R setup
ARG INSTALL_R
RUN if [ "$INSTALL_R" = "true" ]; then \
    apt-get update && apt-get install -y --no-install-recommends \
        r-base \
        r-base-dev \
        libcurl4-openssl-dev \
        libssl-dev \
        libxml2-dev \
        libfreetype6-dev \
        libharfbuzz-dev \
        libfribidi-dev \
        libfontconfig1-dev \
        libcairo2-dev \
        libjpeg-dev \
        libpng-dev \
        libtiff5-dev \
        libgl1-mesa-dev \
        automake \
        make \
        g++ \
        && apt-get clean && rm -rf /var/lib/apt/lists/* \
    ; fi

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app files
COPY . .

# Install R packages if needed
RUN if [ "$INSTALL_R" = "true" ]; then \
    Rscript -e "install.packages('SNFtool', repos='https://cloud.r-project.org')" && \
    Rscript -e "install.packages('remotes', repos='https://cloud.r-project.org')" && \
    Rscript -e "remotes::install_github('Shamir-Lab/NEMO/NEMO')" \
; fi

# Expose the port FastAPI runs on
EXPOSE 8000

# Default command (can be overridden by docker-compose)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
