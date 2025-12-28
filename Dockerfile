# Dockerfile
# Use official slim Python image for smaller size and security
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy and install dependencies first (leverages Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project code
COPY . .

# Expose ports for both services
# 8000: FastAPI backend
# 8501: Streamlit UI
EXPOSE 8000
EXPOSE 8501

# Default command: run the API (most common for production/deployment)
# Will be overridden in docker-compose.yml for the UI service
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]