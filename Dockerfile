# Use a slim Python image to keep the Docker image small
FROM python:3.10-slim

ENV STREAMLIT_DISABLE_WATCHDOG_WARNINGS=true

# Install system dependencies required for building native libraries (e.g. OpenBLAS, CMake)
RUN apt-get update && apt-get install -y \
    git build-essential cmake pkg-config \
    libopenblas-dev libcurl4-openssl-dev \
 && rm -rf /var/lib/apt/lists/*

# Clone and build llama.cpp (optional if you're not using it directly)
RUN git clone https://github.com/ggerganov/llama.cpp /llama.cpp
WORKDIR /llama.cpp
RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DLLAMA_BLAS=ON -DLLAMA_OPENBLAS=ON \
 && cmake --build build --config Release -j$(nproc) \
 && ln -s /llama.cpp/build/bin/* /usr/local/bin/

# Move to app directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Fix: Ensure all required extras are included
RUN pip install --no-cache-dir \
    huggingface_hub[hf_xet] \
    streamlit-extras

# Copy app source files
COPY app/*.py .
COPY data/ /app/data

# Optional: Reduce Docker context issues and avoid copying unnecessary files
# Consider using a .dockerignore file to exclude large directories like __pycache__

# Start Streamlit app
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.fileWatcherType=none"]

