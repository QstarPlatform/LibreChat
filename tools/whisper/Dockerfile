FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy requirements and application files
COPY requirements.txt ./
COPY . ./

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Expose the application port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]