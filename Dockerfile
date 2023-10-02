# Use the Ubuntu as the base image
FROM ubuntu:latest

# Set non-interactive mode during build
ENV DEBIAN_FRONTEND=noninteractive

# Update the package list and install required packages
RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install desired Python packages
RUN pip3 install numpy==1.23.1 pandas==1.4.3 matplotlib==3.7.1 seaborn==0.12.2

# Set the working directory
WORKDIR /app

# Copy all files from the host's current directory to the container's working directory
COPY . /app

# Optionally, remove "acsac275container_latest.tar" if it exists
RUN rm -f /app/acsac275container_latest.tar