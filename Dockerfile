# Use a base Python image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the local package and other files into the container
COPY highway_simulation /app/highway_simulation
COPY main.py /app/main.py
COPY PPO_MODEL.zip /app/PPO_MODEL.zip
COPY requirements.txt /app/requirements.txt

# Install required dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install the local package
RUN pip install /app/highway_simulation


# Define the default command
CMD ["python", "main.py"]
