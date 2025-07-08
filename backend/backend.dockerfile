# Use the official lightweight Python image.
FROM python:3.10-slim

# Set working directory
WORKDIR /app

RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
    fastapi uvicorn[standard] httpx

# Copy your application code
COPY . .

# Expose port 5000
EXPOSE 5000

# Set timezone
ENV TZ=America/Toronto
RUN apt-get update && apt-get install -y tzdata && rm -rf /var/lib/apt/lists/*

# Run Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000", "--reload"]

# docker run --rm -p 5000:5000 [image-name]