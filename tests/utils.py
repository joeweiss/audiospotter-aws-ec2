import boto3
from botocore.stub import Stubber
from io import BytesIO


def return_stubber_client_for_filedownload(bucket_name, key, bcontents=None):
    s3_client = boto3.client(
        "s3",
    )
    stubber = Stubber(s3_client)

    expected_params = {
        "Bucket": bucket_name,
        "Key": key,
    }

    if not bcontents:
        content = b"This is mocked audio content"
        # Get object
        bcontents = BytesIO()
        bcontents.write(content)
        bcontents.seek(0)

    # Head object
    response = {
        "ContentLength": 10,
        "ContentType": "utf-8",
        "ResponseMetadata": {
            "Bucket": bucket_name,
        },
    }
    stubber.add_response("head_object", response, expected_params)

    response = {
        "ContentLength": bcontents.getbuffer().nbytes,
        "ContentType": "utf-8",
        "Body": bcontents,
        "ResponseMetadata": {
            "Bucket": bucket_name,
        },
    }

    stubber.add_response("get_object", response, expected_params)
    stubber.activate()

    return s3_client


def return_stubber_client_for_filedownload_404(bucket_name, key):
    s3_client = boto3.client(
        "s3",
    )
    stubber = Stubber(s3_client)

    stubber.add_client_error("head_object", "NoSuchKey")
    stubber.add_client_error("get_object", "NoSuchKey")

    stubber.activate()

    return s3_client