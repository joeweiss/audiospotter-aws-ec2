from dotenv import load_dotenv
import os
import requests
import tempfile

from remote import Remote

load_dotenv(".env")

API_ENDPOINT = "http://web:8000/api"
API_KEY = "local-key"

S3_ACCESS_KEY = os.environ.get("S3_ACCESS_KEY")
S3_SECRET_KEY = os.environ.get("S3_SECRET_KEY")
RUNNER_COUNT = os.environ.get("RUNNER_COUNT", 4)

INSTANCE_TYPE = "local-type"
INSTANCE_ID = "local-id"

SLEEP_AFTER_EMPTY_QUEUE_SECONDS = 30

PID = os.getpid()


def main():
    with tempfile.TemporaryDirectory() as temp_dir:
        remote = Remote(
            api_endpoint=API_ENDPOINT,
            api_key=API_KEY,
            aws_access_key_id=S3_ACCESS_KEY,
            aws_secret_access_key=S3_SECRET_KEY,
            pid=PID,
            processor_id=INSTANCE_ID,
            processor_type=INSTANCE_TYPE,
            audio_directory=temp_dir,
            runner_count=RUNNER_COUNT,
            shutdown_on_empty_processing_queue=False,
        )
        remote.run_queue()


if __name__ == "__main__":
    main()
