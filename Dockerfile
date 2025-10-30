# Use an official lightweight Python image
FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install system deps for streamlit if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

# Copy application files
COPY . /app

# Expose a default port (CapRover will set $PORT at runtime)
EXPOSE 8080

# Start Streamlit, honoring the $PORT env var set by CapRover (fallback 8080)
CMD ["sh", "-c", "streamlit run 1_Home_Page.py --server.port ${PORT:-80} --server.address 0.0.0.0 --server.headless true"]
