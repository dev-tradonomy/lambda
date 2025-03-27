# FROM public.ecr.aws/lambda/python:3.13

# # Install system dependencies using dnf instead of yum
# RUN dnf update -y && \
#     dnf install -y \
#     zip \
#     postgresql-devel \
#     gcc \
#     python3-devel && \
#     dnf clean all

# # Set working directory
# WORKDIR ${LAMBDA_TASK_ROOT}

# # Copy requirements first to leverage Docker cache
# COPY requirements.txt .

# # Install Python dependencies
# RUN python3 -m pip install --upgrade pip && \
#     mkdir -p ./python && \
#     pip install -r requirements.txt --platform manylinux2014_x86_64 --target ./python --only-binary=:all: && \
#     pip install psycopg2-binary --platform manylinux2014_x86_64 --target ./python --only-binary=:all:

# # Copy function code
# COPY lambda_function.py .

# # Create deployment package
# RUN mkdir -p /tmp/package && \
#     cp -r ${LAMBDA_TASK_ROOT}/python/* /tmp/package/ && \
#     cp lambda_function.py /tmp/package/ && \
#     cd /tmp/package && \
#     zip -r9 /tmp/deployment_package.zip .

# CMD ["lambda_function.lambda_handler"]



FROM python:3.13

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
    zip \
    libpq-dev \
    gcc \
    python3-dev && \
    apt-get clean

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy function code
COPY lambda_function.py .

# Run script as ECS task
CMD ["python", "lambda_function.py"]