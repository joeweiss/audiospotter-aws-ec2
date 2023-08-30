from remote import Remote

from dotenv import load_dotenv
import os

from unittest.mock import patch
from collections import namedtuple
import pytest
import copy

from unittest.mock import MagicMock

load_dotenv(".env")

API_ENDPOINT = os.environ.get("API_ENDPOINT", "")
API_KEY = os.environ.get("API_KEY", "")
S3_ACCESS_KEY = os.environ.get("S3_ACCESS_KEY", "")
S3_SECRET_KEY = os.environ.get("S3_SECRET_KEY", "")

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


# def test_queue():
#     if not LIVE_TEST:
#         return
#     # This only works if the processor_id exist on the api server.
#     remote = Remote(api_endpoint=API_ENDPOINT, api_key=API_KEY, processor_id="local123")
#     item = remote._return_queue_item()
#     pprint(item)
#     assert item == None


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
        expected_queue_response = dict(VALID_QUEUE_RESPONSE)
        mocked_response = Response(
            status_code=200, json=lambda: expected_queue_response
        )
        mocked_queue_response.return_value = mocked_response
        remote = Remote(
            api_endpoint=API_ENDPOINT, api_key=API_KEY, processor_id="local123"
        )
        item = remote._return_queue_item()
        assert "audio" in item
        assert "file_path" in item["audio"]


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
    remote.queued_audio_dict = dict(VALID_QUEUE_RESPONSE)
    remote._retrieve_file()
    assert remote.audio_file_obj != None
    assert remote.audio_filepath == "./soundscape.wav"
    remote._cleanup_files()
    assert os.path.exists(remote.audio_filepath) == False

    # Handle missing file (404 from S3)
    error_audio_dict = dict(remote.queued_audio_dict)
    error_audio_dict["audio"]["file_path"] = (
        remote.queued_audio_dict["audio"]["file_path"] + ".nothere"
    )
    remote.queued_audio_dict = error_audio_dict
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
    remote.queued_audio_dict = dict(VALID_QUEUE_RESPONSE)  # Reset valid audio dict.
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
        "audio": {
            "file_path": key,
            "file_source": {
                "s3_bucket": bucket_name,
            },
        },
        "group": {
            "analyzer_config": {
                "analyzer": {"id": 1, "name": "BirdNET-Analyzer"},
                "minimum_detection_confidence": 0.6,
                "config": {},
                "id": 2,
            },
            "id": 1,
            "name": "Main Group",
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
    remote.queued_audio_dict = dict(queued_audio_dict)
    remote._retrieve_file()  # This is a real file now, let's analyze it.
    remote._analyze_file()
    assert len(remote.recording.detections) == 4
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
        "audio": {
            "file_path": "PROJECT/GROUP/file.flac",
            "file_source": {
                "s3_bucket": "non-existant-bucket",
            },
        }
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
    remote.queued_audio_dict = copy.deepcopy(dict(VALID_QUEUE_RESPONSE_LIVE_ANALYZE))
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

    assert remote.analyzer.version == "2.4"
    assert remote.analyzer.model_download_was_required == False

    pprint(remote._format_results_for_api())

    results = remote._format_results_for_api()
    pprint(results)

    assert results["file_checksum"] == "cfe5e3e09026b622f98c3572f82091f8"
    assert len(results["detections"]) == 25
    assert results["analyzer_version"] == "2.4"

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

    # Test with 2.3.

    # Test live file download.
    remote = Remote(
        api_endpoint=API_ENDPOINT,
        api_key=API_KEY,
        processor_id="local123",
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
    )

    remote.queued_audio_dict = copy.deepcopy(dict(VALID_QUEUE_RESPONSE_LIVE_ANALYZE))

    # Patch response to use 2.3.
    remote.queued_audio_dict["group"]["analyzer_config"]["analyzer"][
        "base_version"
    ] = "2.3"

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

    assert remote.analyzer.version == "2.3"

    pprint(remote._format_results_for_api())

    results = remote._format_results_for_api()
    pprint(results)

    assert results["file_checksum"] == "cfe5e3e09026b622f98c3572f82091f8"
    assert len(results["detections"]) == 12
    assert results["analyzer_version"] == "2.3"

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


def test_live_multi_analyze():
    # Processes queue items that have different analyzer configs with the same Remote.
    # This mimics how the runner works when pulling from a queue with multiple active Analysis Groups.

    queue_item_1 = copy.deepcopy(dict(VALID_QUEUE_RESPONSE_LIVE_ANALYZE))
    queue_item_2 = copy.deepcopy(dict(VALID_QUEUE_RESPONSE_LIVE_ANALYZE))
    queue_item_3 = copy.deepcopy(dict(VALID_QUEUE_RESPONSE_LIVE_ANALYZE))

    # Modify second queue item with different version.
    queue_item_2["group"]["analyzer_config"]["analyzer"]["base_version"] = "2.3"

    assert queue_item_1["group"]["analyzer_config"]["analyzer"]["base_version"] == "2.4"
    assert queue_item_2["group"]["analyzer_config"]["analyzer"]["base_version"] == "2.3"
    assert queue_item_3["group"]["analyzer_config"]["analyzer"]["base_version"] == "2.4"

    if not LIVE_TEST:
        return

    # Create one remote.
    remote = Remote(
        api_endpoint=API_ENDPOINT,
        api_key=API_KEY,
        processor_id="local123",
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
    )

    # Test live file download with queue item 1, which is 2.4.
    remote.queued_audio_dict = queue_item_1

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

    assert remote.analyzer.version == "2.4"
    assert remote.analyzer.model_download_was_required == False

    pprint(remote._format_results_for_api())

    results = remote._format_results_for_api()
    pprint(results)

    assert results["file_checksum"] == "cfe5e3e09026b622f98c3572f82091f8"
    assert len(results["detections"]) == 25
    assert results["analyzer_version"] == "2.4"

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

    # Test live file download with queue item 2, which uses 2.3.
    remote.queued_audio_dict = queue_item_2

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

    assert remote.analyzer.version == "2.3"
    assert remote.analyzer.model_download_was_required == False

    pprint(remote._format_results_for_api())

    results = remote._format_results_for_api()
    pprint(results)

    assert results["file_checksum"] == "cfe5e3e09026b622f98c3572f82091f8"
    assert len(results["detections"]) == 12
    assert results["analyzer_version"] == "2.3"

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

    # Test live file download with queue item 3, which uses 2.4.
    remote.queued_audio_dict = queue_item_3

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

    assert remote.analyzer.version == "2.4"
    assert remote.analyzer.model_download_was_required == False

    pprint(remote._format_results_for_api())

    results = remote._format_results_for_api()
    pprint(results)

    assert results["file_checksum"] == "cfe5e3e09026b622f98c3572f82091f8"
    assert len(results["detections"]) == 25
    assert results["analyzer_version"] == "2.4"

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

    # Ensure that only two Analyzers were created.
    assert len(remote._analyzers.items()) == 2
    # Ensure that only two initializations occurred.
    assert remote._analyzers_init_count == 2


def test_live_species_list_analyze():
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
    remote.queued_audio_dict = dict(VALID_QUEUE_SPECIES_LIST_RESPONSE)
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

    # Confirm custom species list from API is loaded.
    # print(remote.analyzer.custom_species_list)
    assert remote.analyzer.custom_species_list == ["Haemorhous mexicanus_House Finch"]

    results = remote._format_results_for_api()

    assert results["file_checksum"] == "cfe5e3e09026b622f98c3572f82091f8"
    assert len(results["detections"]) == 3

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


def test_live_analyze_custom_classifier():
    if not LIVE_TEST:
        return

    # Test live file download.

    print(API_ENDPOINT)

    remote = Remote(
        api_endpoint=API_ENDPOINT,
        api_key=API_KEY,
        processor_id="local123",
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
    )
    remote.queued_audio_dict = dict(VALID_QUEUE_CUSTOM_CLASSIFIERS_RESPONSE)
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
    assert len(results["detections"]) == 39

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


LIVE_QUEUE_RESPONSE = {
    "id": 3228,
    "status": "in_progress",
    "group": {
        "id": 3228,
        "analyzer_config": {
            "analyzer": {
                "id": 1,
                "name": "BirdNET-Analyzer",
                "model_fp32_file": "/media/BirdNET_GLOBAL_3K_V2.3_Model_FP32.tflite",
                "model_fp16_file": "/media/BirdNET_GLOBAL_3K_V2.3_MData_Model_FP16.tflite",
                "labels_file": "/media/BirdNET_GLOBAL_3K_V2.3_Labels_0eHo4Cy.txt",
            },
            "minimum_detection_confidence": 0.25,
            "minimum_detection_clip_confidence": 0.5,
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
        "file_path": "OLY/OLY_41083/Stn_4/OLY_41083-4_20210824_100000.flac",
        "file_source": {
            "id": 1,
            "name": "Main Bucket",
            "s3_bucket": "birdpop-audio-raw-storage",
            "s3_region": "us-west-1",
            "source_type": "S3",
        },
        "id": 3228,
    },
}


VALID_QUEUE_RESPONSE = {
    "id": 3228,
    "status": "in_progress",
    "group": {
        "id": 3228,
        "analyzer_config": {
            "analyzer": {
                "id": 1,
                "name": "BirdNET-Analyzer",
                "model_fp32_file": "/media/BirdNET_GLOBAL_3K_V2.3_Model_FP32.tflite",
                "model_fp16_file": "/media/BirdNET_GLOBAL_3K_V2.3_MData_Model_FP16.tflite",
                "labels_file": "/media/BirdNET_GLOBAL_3K_V2.3_Labels_0eHo4Cy.txt",
            },
            "minimum_detection_confidence": 0.25,
            "minimum_detection_clip_confidence": 0.5,
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


VALID_QUEUE_RESPONSE_LIVE_ANALYZE = {
    "id": 3228,
    "status": "in_progress",
    "group": {
        "id": 3228,
        "analyzer_config": {
            "analyzer": {"id": 1, "name": "BirdNET-Analyzer", "base_version": "2.4"},
            "minimum_detection_confidence": 0.25,
            "minimum_detection_clip_confidence": 0.5,
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


VALID_QUEUE_SPECIES_LIST_RESPONSE = {
    "id": 3228,
    "status": "in_progress",
    "group": {
        "id": 3228,
        "analyzer_config": {
            "analyzer": {
                "id": 1,
                "name": "BirdNET-Analyzer",
                "base_version": "2.3",
            },
            "minimum_detection_confidence": 0.25,
            "minimum_detection_clip_confidence": 0.5,
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
            "species_list": ["Haemorhous mexicanus_House Finch"],
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


VALID_QUEUE_CUSTOM_CLASSIFIERS_RESPONSE = {
    "id": 3228,
    "status": "in_progress",
    "group": {
        "id": 3228,
        "analyzer_config": {
            "analyzer": {
                "id": 1,
                "name": "BirdNET-Analyzer",
                "model_fp32_file": "/media/Custom_Classifier.tflite",
                "model_fp16_file": "",
                "labels_file": "/media/Custom_Classifier_Labels.txt",
            },
            "minimum_detection_confidence": 0.9,
            "minimum_detection_clip_confidence": 1.0,
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
