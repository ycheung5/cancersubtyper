#!/bin/sh
set -eu

SERVICE_MODE="${1:-api}"
VENV_PATH="/opt/venv"

install_python_deps() {
  requirements_file="$1"
  hash_file="$2"
  requirements_hash="$(sha256sum "$requirements_file" | awk '{print $1}')"

  if [ ! -x "$VENV_PATH/bin/python" ]; then
    python -m venv "$VENV_PATH"
  fi

  . "$VENV_PATH/bin/activate"

  if [ ! -f "$hash_file" ] || [ "$(cat "$hash_file")" != "$requirements_hash" ]; then
    pip install --upgrade pip
    pip install --no-cache-dir -r "$requirements_file"
    printf '%s' "$requirements_hash" > "$hash_file"
  fi
}

install_api_os_deps() {
  apt-get update
  apt-get install -y --no-install-recommends git
  rm -rf /var/lib/apt/lists/*
}

install_worker_os_deps() {
  apt-get update
  apt-get install -y --no-install-recommends \
    git \
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
    g++
  rm -rf /var/lib/apt/lists/*
}

install_r_deps() {
  r_hash_file="/opt/r-libs/.r-deps.sha256"
  r_deps_hash="nemo-r-deps-v1"

  mkdir -p /opt/r-libs

  if [ ! -f "$r_hash_file" ] || [ "$(cat "$r_hash_file")" != "$r_deps_hash" ]; then
    Rscript -e "install.packages('SNFtool', repos='https://cloud.r-project.org')"
    Rscript -e "install.packages('remotes', repos='https://cloud.r-project.org')"
    Rscript -e "remotes::install_github('Shamir-Lab/NEMO/NEMO')"
    printf '%s' "$r_deps_hash" > "$r_hash_file"
  fi
}

case "$SERVICE_MODE" in
  api)
    install_api_os_deps
    install_python_deps "requirements.txt" "$VENV_PATH/.requirements.sha256"
    exec "$VENV_PATH/bin/uvicorn" main:app --host 0.0.0.0 --port 8000 --reload --reload-dir /app
    ;;
  worker)
    install_worker_os_deps
    install_python_deps "requirements_worker.txt" "$VENV_PATH/.requirements-worker.sha256"
    install_r_deps
    exec "$VENV_PATH/bin/celery" -A celery_config worker --loglevel=info --concurrency=2
    ;;
  flower)
    install_python_deps "requirements.txt" "$VENV_PATH/.requirements.sha256"
    exec "$VENV_PATH/bin/celery" -A celery_config flower --broker="$CELERY_BROKER_URL" --port=5555
    ;;
  *)
    echo "Unknown service mode: $SERVICE_MODE" >&2
    exit 1
    ;;
esac
