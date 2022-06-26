# use to build the inference container for running on x86 hardware
# keep in sync with the lt4 version so both can be built in parallel
# Base Image
FROM nvcr.io/nvidia/pytorch:22.05-py3

# Updating repository sources
RUN apt-get update

# Copy Files
COPY requirements.txt requirements.txt

# pip install
RUN pip install --upgrade pip

RUN pip install -r requirements.txt



RUN mkdir -p /inference/api; mkdir -p /var/log/inference
WORKDIR /inference/api
COPY . /inference/api
# CMD [ "python3", "/inference/api/server.py"]
