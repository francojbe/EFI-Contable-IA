FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download model at build time to save startup time (optional but recommended for sentence-transformers)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy application files
COPY . .

# Expose port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8000"]
