from dotenv import load_dotenv
import os


from remote_perch_embed import Remote

load_dotenv(".env")

API_ENDPOINT = os.environ.get("API_ENDPOINT")
API_KEY = os.environ.get("API_KEY")

S3_ACCESS_KEY = os.environ.get("S3_ACCESS_KEY")
S3_SECRET_KEY = os.environ.get("S3_SECRET_KEY")
RUNNER_COUNT = os.environ.get("RUNNER_COUNT", 4)

EMBEDDING_BUCKET = os.environ.get("EMBEDDING_BUCKET")

INSTANCE_TYPE = "local-type"
INSTANCE_ID = "local-id"

SLEEP_AFTER_EMPTY_QUEUE_SECONDS = 30

PID = os.getpid()


def main():
    raw_audio_dir = "raw_audio"
    os.makedirs(raw_audio_dir, exist_ok=True)

    remote = Remote(
        api_endpoint=API_ENDPOINT,
        api_key=API_KEY,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
        pid=PID,
        processor_id=INSTANCE_ID,
        processor_type=INSTANCE_TYPE,
        audio_directory=raw_audio_dir,
        runner_count=RUNNER_COUNT,
        shutdown_on_empty_processing_queue=False,
        location_id=210,  # Magical for the moment.
        limit=10,
        destination_bucket=EMBEDDING_BUCKET,
    )
    print(remote)
    remote._return_audio_items_for_location()

    print(len(remote.audio_list))

    remote._retrieve_files()

    remote._sync_embeddings_from_s3()

    remote._extract_embeddings()

    remote._upload_embeddings()

    remote._shutdown()


if __name__ == "__main__":
    main()
