#!/bin/bash

# Update system packages
sudo yum update -y

# Install Docker
sudo yum install -y docker
sudo service docker start
sudo usermod -a -G docker ubuntu
sudo chkconfig docker on

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose


# Create nginx directory if it doesn't exist
mkdir -p nginx
sudo cp ~/app/nginx/nginx.conf /etc/nginx/nginx.conf

# Pull the latest code and deploy
git pull origin main
sudo docker-compose down
sudo docker build --tag flask-app .
sudo docker run -d -p 5000:5000 flask-app
sudo docker-compose up -d

# Clean up unused images
sudo docker image prune -f