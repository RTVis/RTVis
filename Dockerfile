# Use an official Python runtime as a parent image
FROM python:3.8

# Set the working directory in the container to /app
WORKDIR /rtvis

# Copy the current directory contents into the container at /app
COPY . /rtvis

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8050 available to the world outside this container
EXPOSE 8989

# Run app.py when the container launches
CMD ["python", "-u", "app.py"]
