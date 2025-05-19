FROM python:3.8-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p FlowAlgorithm/output \
    && mkdir -p Predictions/data \
    && mkdir -p Predictions/RegressionModel/models \
    && mkdir -p Predictions/TransformerModel/models

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Default command to run when the container starts
CMD ["python", "-m", "FlowAlgorithm.main"]

# Expose ports for Streamlit apps
EXPOSE 8501
