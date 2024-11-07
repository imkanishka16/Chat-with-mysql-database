# # Use Python base image
# FROM python:3.12

# # Install system dependencies
# RUN apt-get update && apt-get install -y \
#     default-jdk \
#     libgl1-mesa-glx \
#     libglib2.0-0 \
#     && rm -rf /var/lib/apt/lists/*

# # # Add arguments for all environment variables
# # ARG OPENAI_API_KEY
# # ARG DB_HOST
# # ARG DB_USER
# # ARG DB_PASSWORD
# # ARG DB_NAME

# # # Set environment variables
# # ENV OPENAI_API_KEY=$OPENAI_API_KEY
# # ENV DB_HOST=$DB_HOST
# # ENV DB_USER=$DB_USER
# # ENV DB_PASSWORD=$DB_PASSWORD
# # ENV DB_NAME=$DB_NAME
    
# # Set working directory
# WORKDIR /flask-app

# # Copy requirements first to leverage Docker cache
# COPY requirements.txt requirements.txt

# # Install Python dependencies
# RUN pip install --no-cache-dir -r requirements.txt

# # Copy the rest of the application
# COPY . .

# # Expose port
# EXPOSE 5001

# # Command to run the application
# CMD ["python", "new.py"]


# Use Python base image
FROM python:3.12

# Install system dependencies
RUN apt-get update && apt-get install -y \
    default-jdk \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Add arguments for all environment variables
ARG OPENAI_API_KEY
ARG DB_HOST
ARG DB_USER
ARG DB_PASSWORD
ARG DB_NAME

# Set environment variables
ENV OPENAI_API_KEY=$OPENAI_API_KEY
ENV DB_HOST=$DB_HOST
ENV DB_USER=$DB_USER
ENV DB_PASSWORD=$DB_PASSWORD
ENV DB_NAME=$DB_NAME
    
# Set working directory
WORKDIR /flask-app

# Copy requirements first to leverage Docker cache
COPY requirements.txt requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose port
EXPOSE 5000

# Command to run the application
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "new:app"]