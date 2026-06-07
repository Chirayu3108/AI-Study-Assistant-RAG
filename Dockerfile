# Use a slim Python 3.11 base image to keep the container small
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies needed by some loaders (e.g. PyMuPDF, unstructured)
# libgl1 replaces libgl1-mesa-glx which was removed in Debian Trixie
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first so Docker can cache this layer separately
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project into the container
COPY . .

# Create the data directories in case they don't exist
RUN mkdir -p data/vector_store data/pdf_files data/text_files

# Expose the default Streamlit port
EXPOSE 8501

# Tell Streamlit not to open a browser and to listen on all network interfaces
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Launch the app
CMD ["streamlit", "run", "app.py"]
