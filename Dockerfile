# Use official Python image
FROM python:3.12-slim

# Prevent Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1

# Prevent Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1

# Set working directory inside container
WORKDIR /app

# Copy dependency file first (better caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install torch torchvision torchaudio
RUN python -m spacy download en_core_web_sm


# Copy rest of the application
COPY . .

# Expose port (change if needed)
EXPOSE 5050

# Run the application
CMD ["python", "run.py"]