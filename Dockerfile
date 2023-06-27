# Use an official Python runtime as a parent image
FROM python:3.8

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHON UNBUFFERED 1

# Set the working directory in the container to /app
WORKDIR /rtvis

# Copy the current directory contents into the container at /app
COPY . /rtvis

RUN echo "networks:\n  rtvis_net:\n    ipv4_address: 173.26.0.2\n\nnetworks:\n  rtvis_net:\n    external: true" >> docker-compose.yml

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
# RUN python dataprocessing.py

# Make port 8050 available to the world outside this container
EXPOSE 8050

# Run app.py when the container launches
CMD ["gunicorn", "--workers=1", "--threads=1", "-b", "0.0.0.0:8050", "app:server"]
