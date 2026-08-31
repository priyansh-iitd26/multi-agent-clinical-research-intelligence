# docker file to create the docker image

# using multi-stage build in docker primarily to drastically 
# reduce the size of the final image and improve security by 
# separating the build environment from the runtime environment
# https://docs.docker.com/build/building/multi-stage/


# STAGE 1: BUILDER
# this stage installs all python dependencies
# it is called builder - build tools
# installing massive tools that compile code

# base image
# small python 3.12 linux image
FROM python:3.12-slim AS builder

# working dir
# setting /app as the working dir inside the container
WORKDIR /app

# installing build time linux packages system dependencies needed to compile some python packages
# some python packages (like asyncpg) include C extensions
# that need to be compiled from source - requires build tools

# gcc -> C/C++ compiler
# libpq-dev -> PostgreSQL headers/libraries
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# creating a virtual environment to hold all installed packages
RUN python -m venv /opt/venv

# activating the virtual env
ENV PATH="/opt/venv/bin:$PATH"

# copy
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


# STAGE 2: RUNTIME
# this stage creates the actual image that runs on Cloud Run
# it copies the build tools from builder stage
# creates runtime tools

# base image
FROM python:3.12-slim
# this image does NOT have build tools from stage-1 yet (gcc, libpq-dev)
# much smaller and more secure than carrying build tools into production

# working dir
WORKDIR /app

# installing runtime dependencies 
# curl for healthcheck, libpq5 for PostgreSQL's runtime client library
RUN apt-get update && apt-get install -y \
    curl \
    libpq5 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# copying virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# activating the virtual environment in this stage
ENV PATH="/opt/venv/bin:$PATH"

# copying the application source code into the container
COPY config/       ./config/
COPY ingestion/    ./ingestion/
COPY processing/   ./processing/
COPY memory/       ./memory/
COPY agents/       ./agents/
COPY tools/        ./tools/
COPY graph/        ./graph/
COPY api/          ./api/
COPY pyproject.toml .
# we copy each directory explicitly rather than COPY . .
# (which would copy everything including .env, .git, __pycache__ etc.)
# explicit copies => only what the application actually needs

ENV PYTHONPATH="/app"
# tells python where to find the custom modules


# runtime configuration

ENV PYTHONUNBUFFERED=1
# PYTHONUNBUFFERED=1 tells python not to buffer stdout/stderr output
# without this, log lines might get buffered and appear all at once
# instead of streaming in real time
# Cloud Run captures stdout for logs - we need real-time streaming
# so we can see log lines in the GCP Console as they happen

ENV PYTHONDONTWRITEBYTECODE=1
# PYTHONDONTWRITEBYTECODE=1 tells the python interpreter not to create
# .pyc files or __pycache__ directories
# done to save temporary memory in docker container

# port 8000
EXPOSE 8000

# health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# HEALTHCHECK tells docker how to monitor the application to ensure
# that it's functionally running, rather than just checking
# if the container is running

# --interval=30s -> docker will wait exactly 30 sec between each health check

# --timeout=10s -> when docker runs the `curl` command, gives the app 10 seconds
#                  to respond, otherwise marked as failure by docker

# --start-period=40s -> grace period when the container first boots up
#                      (gives the FastAPI app time to initialize)

# --retries=3 -> docker flags the container unhealthy only after 3 consecutive failures

# curl -f http://localhost:8000/api/v1/health || exit 1
# this uses the curl tool to make an actual HTTP GET request to the API's health endpoint


# START CMD
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]