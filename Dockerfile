FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# Copy application code
COPY backend/ /app/backend/
COPY IMC_v2_Interstellar_Mission_Compiler.html /app/

# Create data and output directories
RUN mkdir -p /app/backend/data /app/backend/output

# Initialize the stellar catalog database
RUN python -c "import sys; sys.path.insert(0, '/app/backend'); from imc.catalog import create_catalog_db; create_catalog_db('/app/backend/data/local_bubble.db')"

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
