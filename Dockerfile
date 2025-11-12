# -------------- Base image --------------
FROM python:3.12-slim AS base

# Prevent Python from writing .pyc files / buffer stdout
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONWARNINGS="ignore" \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System deps (certs, curl used for health/debug)
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

# -------------- App setup --------------
WORKDIR /app

# Copy dependency list 
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy  application code
COPY . /app

# Create folders 
RUN mkdir -p /app/raw /app/result

# For security, run as a non-root user
RUN useradd -m appuser
USER appuser

# -------------- Runtime --------------

EXPOSE 8800
CMD ["python", "app.py"]
