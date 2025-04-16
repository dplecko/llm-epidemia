FROM rocker/r-ver:4.3.2

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    libharfbuzz-dev \
    libfribidi-dev \
    libfontconfig1-dev \
    libfreetype6-dev \
    libtiff5-dev \
    libjpeg-dev \
    libpng-dev \
    libblas-dev \
    cmake \
    && apt-get clean

# Install R packages
RUN R -e "install.packages(c('data.table', 'mice', 'fst', 'arrow', 'haven', 'remotes'), repos = 'https://cloud.r-project.org')"
RUN R -e "remotes::install_github('eth-mds/ricu')"

# Copy local R code and config
COPY . /app
WORKDIR /app

# Default command
CMD ["Rscript", "nhanes.R"]

