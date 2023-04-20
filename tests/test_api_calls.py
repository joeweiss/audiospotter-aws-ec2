from remote import Remote

from dotenv import load_dotenv
import os

from unittest.mock import patch
from collections import namedtuple
import pytest

load_dotenv(".env")

API_ENDPOINT = os.environ.get("API_ENDPOINT")
API_KEY = os.environ.get("API_KEY")
S3_ACCESS_KEY = os.environ.get("S3_ACCESS_KEY")
S3_SECRET_KEY = os.environ.get("S3_SECRET_KEY")

from birdnetlib.analyzer import Analyzer
from io import BytesIO
from pprint import pprint

import boto3

from .utils import (
    return_stubber_client_for_filedownload,
    return_stubber_client_for_filedownload_404,
)

s3 = boto3.resource("s3")

LIVE_TEST = False


def test_queue():
    if not LIVE_TEST:
        return
    remote = Remote(api_endpoint=API_ENDPOINT, api_key=API_KEY, processor_id="local123")
    item = remote._return_queue_item()
    pprint(item)
    assert item == None


def test_mocked_queue_request():
    # Test empty queue response.
    with patch("remote.requests.post") as mocked_queue_response:
        Response = namedtuple("Response", ["status_code", "json"])
        expected_queue_response = {}
        mocked_response = Response(
            status_code=200, json=lambda: expected_queue_response
        )
        mocked_queue_response.return_value = mocked_response
        remote = Remote(
            api_endpoint=API_ENDPOINT, api_key=API_KEY, processor_id="local123"
        )
        item = remote._return_queue_item()
        assert item == None

    # Test endpoint verification for https.
    remote = Remote(
        api_endpoint="https://example.com", api_key=API_KEY, processor_id="local123"
    )
    assert remote.verify_request

    # Test endpoint verification for http.
    remote = Remote(
        api_endpoint="http://example.com", api_key=API_KEY, processor_id="local123"
    )
    assert remote.verify_request == False

    # Test non-200 response.
    # TODO: Handle timeouts and retries.
    with patch("remote.requests.post") as mocked_queue_response:
        Response = namedtuple("Response", ["status_code", "json"])
        expected_queue_response = {}
        mocked_response = Response(
            status_code=404, json=lambda: expected_queue_response
        )
        mocked_queue_response.return_value = mocked_response
        remote = Remote(
            api_endpoint=API_ENDPOINT, api_key=API_KEY, processor_id="local123"
        )
        expected_error_text = "Remote could not connect to API endpoint (status 404)."
        with pytest.raises(ConnectionError) as e:
            item = remote._return_queue_item()
        assert str(e.value) == expected_error_text

    # Test queue item response.
    with patch("remote.requests.post") as mocked_queue_response:
        Response = namedtuple("Response", ["status_code", "json"])
        expected_queue_response = VALID_QUEUE_RESPONSE
        mocked_response = Response(
            status_code=200, json=lambda: expected_queue_response
        )
        mocked_queue_response.return_value = mocked_response
        remote = Remote(
            api_endpoint=API_ENDPOINT, api_key=API_KEY, processor_id="local123"
        )
        item = remote._return_queue_item()
        assert "file_path" in item


def test_live_download():
    if not LIVE_TEST:
        return

    # Test live file download.
    remote = Remote(
        api_endpoint=API_ENDPOINT,
        api_key=API_KEY,
        processor_id="local123",
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
    )
    remote.queued_audio_dict = VALID_QUEUE_RESPONSE.copy()
    remote._retrieve_file()
    assert remote.audio_file_obj != None
    assert remote.audio_filepath == "./soundscape.wav"
    remote._cleanup_files()
    assert os.path.exists(remote.audio_filepath) == False

    # Handle missing file (404 from S3)
    remote.queued_audio_dict["file_path"] = (
        remote.queued_audio_dict["file_path"] + ".nothere"
    )
    expected_error_text = (
        "Remote could not find audio file on S3 "
        + "(error: An error occurred (404) when calling the HeadObject operation: Not Found)."
    )
    with pytest.raises(ConnectionError) as e:
        remote._retrieve_file()
    assert str(e.value) == expected_error_text
    assert remote.audio_file_obj == None
    # Confirm files are cleaned up automatically on exceptions.
    assert os.path.exists(remote.audio_filepath) == False

    # Handle credential error (403 from S3)
    remote.queued_audio_dict = VALID_QUEUE_RESPONSE.copy()  # Reset valid audio dict.
    # Reset remote with non-valid credentials.
    remote._client = None
    remote.aws_access_key_id = remote.aws_access_key_id + "Z"
    expected_error_text = (
        "Remote could not find audio file on S3 "
        + "(error: An error occurred (403) when calling the HeadObject operation: Forbidden)."
    )
    with pytest.raises(ConnectionError) as e:
        remote._retrieve_file()
    assert str(e.value) == expected_error_text
    assert remote.audio_file_obj == None
    # Confirm files are cleaned up automatically on exceptions.
    assert os.path.exists(remote.audio_filepath) == False


def test_mocked_downloads():
    # Setup stubber.
    bucket_name = "non-existant-bucket"
    key = "PROJECT/GROUP/file.wav"
    with open("tests/test_files/soundscape.wav", "rb") as fh:
        buf = BytesIO(fh.read())
    mocked_s3_client = return_stubber_client_for_filedownload(
        bucket_name, key, bcontents=buf
    )
    # Fake the queue return.
    queued_audio_dict = {
        "file_path": key,
        "file_source": {
            "s3_bucket": bucket_name,
        },
        "project": {
            "analyzer_config": {
                "analyzer": {"id": 1, "name": "BirdNET-Analyzer"},
                "config": {"minimum_detection_confidence": 0.6},
                "id": 2,
            },
            "id": 1,
            "name": "Main Project",
        },
    }

    # Test file download.
    remote = Remote(
        api_endpoint=API_ENDPOINT,
        api_key=API_KEY,
        processor_id="local123",
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
    )
    remote._client = mocked_s3_client
    remote.queued_audio_dict = queued_audio_dict.copy()
    remote._retrieve_file()  # This is a real file now, let's analyze it.
    remote._analyze_file()
    assert len(remote.recording.detections) == 2
    assert remote.audio_file_obj != None
    assert remote.audio_filepath == "./file.wav"
    remote._cleanup_files()
    assert os.path.exists(remote.audio_filepath) == False


def test_mocked_s3_error_download():
    # Missing file.
    bucket_name = "non-existant-bucket"
    key = "PROJECT/GROUP/file.wav"
    mocked_s3_client = return_stubber_client_for_filedownload_404(bucket_name, key)

    # Test missing file download.
    remote = Remote(
        api_endpoint=API_ENDPOINT,
        api_key=API_KEY,
        processor_id="local123",
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
    )
    remote._client = mocked_s3_client
    remote.queued_audio_dict = {
        "file_path": "PROJECT/GROUP/file.flac",
        "file_source": {
            "s3_bucket": "non-existant-bucket",
        },
    }

    expected_error_text = (
        "Remote could not find audio file on S3 "
        + "(error: An error occurred (NoSuchKey) when calling the HeadObject operation: )."
    )
    with pytest.raises(ConnectionError) as e:
        remote._retrieve_file()
    assert str(e.value) == expected_error_text
    assert remote.audio_file_obj == None
    # Confirm files are cleaned up automatically on exceptions.
    assert os.path.exists(remote.audio_filepath) == False

    # TODO: Report error back to manager API in a different test.


def test_live_analyze():
    if not LIVE_TEST:
        return

    # Test live file download.
    remote = Remote(
        api_endpoint=API_ENDPOINT,
        api_key=API_KEY,
        processor_id="local123",
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
    )
    remote.queued_audio_dict = VALID_QUEUE_RESPONSE.copy()
    pprint(remote.queued_audio_dict)
    remote._retrieve_file()
    assert remote.audio_file_obj != None
    assert remote.audio_filepath == "./soundscape.wav"

    remote._analyze_file()

    remote._extract_detections_as_audio()
    remote._extract_detections_as_spectrogram()
    remote._upload_extractions()
    remote._upload_json()

    pprint(remote.recording.detections)

    remote._cleanup_files()

    pprint(remote._format_results_for_api())

    results = remote._format_results_for_api()

    assert results["file_checksum"] == "cfe5e3e09026b622f98c3572f82091f8"
    assert len(results["detections"]) == 12

    with patch("remote.requests.post") as mocked_queue_response:
        Response = namedtuple("Response", ["status_code", "json"])
        expected_queue_response = {"id": remote.queued_audio_dict["id"]}
        mocked_response = Response(
            status_code=201, json=lambda: expected_queue_response
        )
        mocked_queue_response.return_value = mocked_response
        result = remote._save_results_to_server()
        assert result is not None

    assert os.path.exists(remote.audio_filepath) == False


def test_live_process():
    if not LIVE_TEST:
        return

    # Mocks interactions with api, but uses Tungite's live test S3 setup.
    with patch("remote.requests.get") as mocked_queue_response, patch(
        "remote.requests.post"
    ) as mocked_post_response:
        # Get response
        Response = namedtuple("Response", ["status_code", "json"])
        expected_queue_response = VALID_QUEUE_RESPONSE.copy()
        mocked_queue_response.return_value = Response(
            status_code=200, json=lambda: expected_queue_response
        )

        # Post response
        expected_report_response = {}
        mocked_post_response.return_value = Response(
            status_code=201, json=lambda: expected_report_response
        )

        remote = Remote(
            api_endpoint=API_ENDPOINT,
            api_key=API_KEY,
            processor_id="local123",
            aws_access_key_id=S3_ACCESS_KEY,
            aws_secret_access_key=S3_SECRET_KEY,
        )

        remote.process()


LIVE_QUEUE_RESPONSE = {
    "file_path": "OLY/OLY_41083/Stn_4/OLY_41083-4_20210824_100000.flac",
    "file_source": {
        "id": 1,
        "name": "Main Bucket",
        "s3_bucket": "birdpop-audio-raw-storage",
        "s3_region": "us-west-1",
        "source_type": "S3",
    },
    "id": 3228,
    "project": {
        "analyzer_config": {
            "analyzer": {"id": 1, "name": "BirdNET-Analyzer"},
            "config": {"minimum_detection_confidence": 0.25},
            "id": 2,
        },
        "id": 1,
        "name": "Main Project",
    },
    "status": "in_progress",
}


VALID_QUEUE_RESPONSE = {
    "file_path": "PROJECT_SLUG/GROUP/soundscape.wav",
    "file_source": {
        "id": 1,
        "name": "Main Bucket",
        "s3_bucket": "birdnet-lib-aws-runner-audio-storage",
        "s3_region": "us-west-1",
        "source_type": "S3",
    },
    "id": 3228,
    "project": {
        "analyzer_config": {
            "analyzer": {"id": 1, "name": "BirdNET-Analyzer"},
            "config": {
                "minimum_detection_confidence": 0.25,
                "minimum_detection_clip_confidence": 0.5,
            },
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
        "id": 1,
        "name": "Main Project",
    },
    "status": "in_progress",
}
