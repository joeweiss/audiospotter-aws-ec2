from dotenv import load_dotenv
import os
import requests
import tempfile

from remote import Remote

load_dotenv(".env")

API_ENDPOINT = os.environ.get("API_ENDPOINT")
API_KEY = os.environ.get("API_KEY")

S3_ACCESS_KEY = os.environ.get("S3_ACCESS_KEY")
S3_SECRET_KEY = os.environ.get("S3_SECRET_KEY")
RUNNER_COUNT = os.environ.get("RUNNER_COUNT", 4)

response = requests.get("http://169.254.169.254/latest/meta-data/instance-type")
INSTANCE_TYPE = response.text

response = requests.get("http://169.254.169.254/latest/meta-data/instance-id")
INSTANCE_ID = response.text

SLEEP_AFTER_EMPTY_QUEUE_SECONDS = 30


def main():
    with tempfile.TemporaryDirectory() as temp_dir:
        remote = Remote(
            api_endpoint=API_ENDPOINT,
            api_key=API_KEY,
            aws_access_key_id=S3_ACCESS_KEY,
            aws_secret_access_key=S3_SECRET_KEY,
            processor_id=INSTANCE_ID,
            processor_type=INSTANCE_TYPE,
            audio_directory=temp_dir,
            runner_count=RUNNER_COUNT,
            shutdown_on_empty_processing_queue=True,
        )
        remote.run_queue()


if __name__ == "__main__":
    main()
