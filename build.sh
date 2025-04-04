docker build -t lambda_function .
docker create --name lambda-container lambda_function
docker cp lambda-container:/tmp/deployment_package.zip ./deployment_package-027.zip
docker rm lambda-container


aws s3 cp deployment_package-027.zip s3://lambda-function-twitter-notification/

aws lambda update-function-code --function-name TwitterBasedNotification --s3-bucket lambda-function-twitter-notification --s3-key deployment_package-027.zip
