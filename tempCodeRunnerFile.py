def lambda_handler(event, context):
#     """AWS Lambda synchronous entry point."""
    return asyncio.run(async_lambda_handler(event, context))