#!/bin/bash

# Update system packages
sudo apt update
sudo apt upgrade -y

# Install Docker
sudo apt install -y docker.io
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -a -G docker ubuntu

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Install nginx if not already installed
sudo apt install -y nginx

# Create nginx directory if it doesn't exist
mkdir -p nginx
# Ensure the source directory exists before copying
if [ -d "~/app/nginx" ]; then
    sudo cp ~/app/nginx/nginx.conf /etc/nginx/nginx.conf
else
    echo "Warning: ~/app/nginx directory not found"
fi

# Pull the latest code and deploy
# Check if git repository exists
if [ -d ".git" ]; then
    git pull origin main
else
    echo "Warning: Not a git repository"
fi

# Docker operations
# Check if docker-compose file exists
if [ -f "docker-compose.yml" ]; then
    sudo docker-compose down
else
    echo "Warning: docker-compose.yml not found"
fi

# Build and run Docker container
if [ -f "Dockerfile" ]; then
    sudo docker build --tag flask-app .
    sudo docker run -d -p 5000:5000 flask-app
else
    echo "Warning: Dockerfile not found"
fi

# Run docker-compose if file exists
if [ -f "docker-compose.yml" ]; then
    sudo docker-compose up -d
fi

# Clean up unused images
sudo docker image prune -f