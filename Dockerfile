# Use a lightweight Python image
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Copy all project files into the container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the correct port for Render (10000)
EXPOSE 10000

# Start the Flask app with Gunicorn for better performance
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "app:app"]
