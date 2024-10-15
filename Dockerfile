# Use the official Python image from Docker Hub
FROM python:3.8-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install Streamlit
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application files
COPY . .

# Expose the port for Streamlit
EXPOSE 8501

# Command to run the Streamlit app