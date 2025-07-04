FROM public.ecr.aws/lambda/python:3.13

# Install system dependencies using dnf instead of yum
RUN dnf update -y && \
    dnf install -y \
    zip \
    postgresql-devel \
    gcc \
    python3-devel && \
    dnf clean all

# Set working directory
WORKDIR ${LAMBDA_TASK_ROOT}

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN python3 -m pip install --upgrade pip && \
    mkdir -p ./python && \
    pip install -r requirements.txt --target ./python

# Copy function code
COPY lambda_function.py .

# Create deployment package
RUN mkdir -p /tmp/package && \
    cp -r ${LAMBDA_TASK_ROOT}/python/* /tmp/package/ && \
    cp lambda_function.py /tmp/package/ && \
    cd /tmp/package && \
    zip -r9 /tmp/deployment_package.zip .

CMD ["lambda_function.lambda_handler"]

