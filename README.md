# lambda

*Step 1: Build and Export Lambda Zip*
```bash
docker build -t lambda_function .
docker create --name lambda-container lambda_function
docker cp lambda-container:/tmp/deployment_package.zip ./deployment_package_dummy-001.zip
docker rm lambda-container
```

*Step 2: Upload to S3*
```bash
aws s3 cp deployment_package_shreyas-001.zip s3://lambda-function-twitter-notification/
```

*Step 3: Update Lambda Function Code*
```bash
aws lambda update-function-code \
  --function-name TwitterBasedNotification \
  --s3-bucket lambda-function-twitter-notification \
  --s3-key deployment_package_shreyas-001.zip
```