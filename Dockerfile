
# Use a lightweight official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy all files to the container
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Start the Flask app with gunicorn
CMD ["gunicorn", "app:app"]
