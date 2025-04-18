FROM nvcr.io/nvidia/pytorch:24.11-py3

# Setup: update apt and install Python dependencies.
RUN apt-get update && apt-get install -y \
      python3-pip \
      python3-venv

RUN pip install --upgrade pip setuptools==69.5.1

# Install Python dependencies.
RUN pip install \
      datasets \
      transformers \
      accelerate \
      wandb \
      dacite \
      pyyaml \
      numpy \
      packaging \
      safetensors \
      tqdm \
      sentencepiece \
      tensorboard \
      pandas \
      jupyter \
      deepspeed \
      seaborn \
      timm

# Install R and R development packages.
RUN apt-get update && apt-get install -y \
      r-base \
      r-base-dev

# (Optional) Install additional R packages.
# RUN Rscript -e "install.packages('tidyverse', repos='http://cran.rstudio.com/')"

# Create a work directory.
RUN mkdir -p /workspace
