from remote import Remote
from dotenv import load_dotenv
import os
import copy
import csv
from pprint import pprint
from birdnetlib import LargeRecording
from birdnetlib.analyzer import Detection
from birdnetlib.exceptions import (
    AudioFormatError,
    IncompatibleAnalyzerError,
)
import librosa
import audioread
import numpy as np
import tensorflow as tf

from datetime import datetime
import tensorflow_hub as hub

load_dotenv(".env")

API_ENDPOINT = os.environ.get("API_ENDPOINT", "")
API_KEY = os.environ.get("API_KEY", "")
S3_ACCESS_KEY = os.environ.get("S3_ACCESS_KEY", "")
S3_SECRET_KEY = os.environ.get("S3_SECRET_KEY", "")


def test_calls():
    assert 1 == 1

    # Test live file download.
    remote = Remote(
        api_endpoint=API_ENDPOINT,
        api_key=API_KEY,
        processor_id="local123",
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
    )

    remote.queued_audio_dict = copy.deepcopy(
        dict(VALID_QUEUE_RESPONSE_LIVE_PERCH_ANALYZE)
    )

    remote._retrieve_file()
    assert remote.audio_file_obj is not None
    assert remote.audio_filepath == "./soundscape.wav"

    # Patch response to use 2.3.
    remote.queued_audio_dict["group"]["analyzer_config"]["analyzer"]["base_version"] = (
        "2.3"
    )

    pprint(remote.queued_audio_dict)
    remote._retrieve_file()
    assert remote.audio_file_obj is not None
    assert remote.audio_filepath == "./soundscape.wav"

    remote._analyze_file()
    remote._extract_detections_as_audio()
    remote._extract_detections_as_spectrogram()
    remote._save_embeddings()
    remote._upload_extractions()
    remote._upload_json()
    remote._upload_embeddings()

    pprint(remote.recording.detections)
    pprint(remote._format_results_for_api())

    remote._cleanup_files()


VALID_QUEUE_RESPONSE_LIVE_PERCH_ANALYZE = {
    "id": 3228,
    "status": "in_progress",
    "group": {
        "id": 3228,
        "analyzer_config": {
            "analyzer": {
                "id": 1,
                "name": "Perch",
                "base_type": "perch",
                "base_version": "8",
            },
            "minimum_detection_confidence": 0.5,
            "minimum_detection_clip_confidence": 0.5,
            "include_embeddings": True,
            "config": {},
            "id": 2,
            "extraction_audio_file_destination": {
                "id": 2,
                "name": "Extraction Bucket",
                "s3_bucket": "birdnet-lib-aws-runner-extraction-storage",
                "s3_region": "us-west-1",
                "source_type": "S3",
            },
            "extraction_spectrogram_file_destination": {
                "id": 2,
                "name": "Extraction Bucket",
                "s3_bucket": "birdnet-lib-aws-runner-extraction-storage",
                "s3_region": "us-west-1",
                "source_type": "S3",
            },
            "analysis_json_file_destination": {
                "id": 3,
                "name": "Data Bucket",
                "s3_bucket": "birdnet-lib-aws-runner-data-storage",
                "s3_region": "us-west-1",
                "source_type": "S3",
            },
        },
        "name": "Main Group",
    },
    "audio": {
        "file_path": "PROJECT_SLUG/GROUP/soundscape.wav",
        "file_source": {
            "id": 1,
            "name": "Main Bucket",
            "s3_bucket": "birdnet-lib-aws-runner-audio-storage",
            "s3_region": "us-west-1",
            "source_type": "S3",
        },
        "id": 3228,
    },
}
