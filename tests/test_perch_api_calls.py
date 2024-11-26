from remote import Remote
from dotenv import load_dotenv
import os
import copy
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
    remote = PerchRemote(
        api_endpoint=API_ENDPOINT,
        api_key=API_KEY,
        processor_id="local123",
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
    )

    remote.queued_audio_dict = copy.deepcopy(dict(VALID_QUEUE_RESPONSE_LIVE_ANALYZE))

    # Patch response to use 2.3.
    remote.queued_audio_dict["group"]["analyzer_config"]["analyzer"]["base_version"] = (
        "2.3"
    )

    pprint(remote.queued_audio_dict)
    remote._retrieve_file()
    assert remote.audio_file_obj is not None
    assert remote.audio_filepath == "./soundscape.wav"

    remote._analyze_file()
    pprint(remote._format_results_for_api())


from birdnetlib.utils import read_audio_segments


class PerchAnalyzer:
    def __init__(self):
        # Load the model using TFHub Lib.
        self.model = hub.load(
            "https://www.kaggle.com/models/google/bird-vocalization-classifier/TensorFlow2/bird-vocalization-classifier/8"
        )
        self.custom_species_list = []  # Not yet implemented.
        self.version = "8"

    def analyze_recording(self, recording):
        # print("analyze_recording, large mode", recording.filename)

        start = 0
        end = recording.sample_secs
        results = {}

        # Read segments via generator function so that the entire audio file is never loaded into RAM.
        # TODO: Adapt this to be used by all Analyzers, assuming this works well with Canopy testing.

        sr = 32000

        for segment in read_audio_segments(recording.path, sr=sr, segment_duration=5):
            c = segment["segment"]
            if len(c) < recording.sample_secs * sr:
                # If below the minimum segment duration, continue.
                del c
                continue
            start = segment["start_sec"]
            end = segment["end_sec"]

            print(start, end)

            # Run the model, check the output.
            model_outputs = self.model.infer_tf(c[np.newaxis, :])

            # Examine various logits.
            print(model_outputs["label"].shape)
            print(model_outputs["genus"].shape)
            print(model_outputs["family"].shape)
            print(model_outputs["order"].shape)

            # Examine the embeddings.
            print(model_outputs["embedding"].shape)

            # Convert logits to probabilities
            logits = model_outputs["label"]
            probabilities = tf.nn.softmax(logits, axis=-1)

            # Get the predicted class index
            predicted_index = np.argmax(probabilities)

            # Get the probability (confidence) of the predicted label
            confidence = probabilities[0, predicted_index]

            print("predicted_index", predicted_index)
            print("Confidence in prediction:", confidence.numpy())

            # Store results
            results[str(start) + "-" + str(end)] = [
                ("something_new", float(confidence))
            ]

            # Clean up.
            del c

        self.results = results
        recording.detection_list = self.detections

    @property
    def detections(self):
        detections = []
        for key, value in self.results.items():
            print(f"{key} -----")
            print(value)
            start_time = float(key.split("-")[0])
            end_time = float(key.split("-")[1])
            print(start_time, end_time)
            for c in value:
                confidence = float(c[1])
                label = c[0]
                scientific_name = label.split("_")[0]
                common_name = label.split("_")[1]
                # print(c[0], f"{c[1]:1.4f}")
                d = Detection(start_time, end_time)
                d.common_name = common_name
                d.scientific_name = scientific_name
                d.confidence = confidence
                d.label = label
                # print(d.as_dict)
                detections.append(d)

        return detections


class PerchLargeRecording(LargeRecording):
    def __init__(
        self,
        analyzer,
        path,
        week_48=-1,
        date=None,
        sensitivity=1,
        lat=None,
        lon=None,
        min_conf=0.1,
        overlap=0,
        return_all_detections=False,
    ):
        super().__init__(
            analyzer,
            path,
            week_48,
            date,
            sensitivity,
            lat,
            lon,
            min_conf,
            overlap,
            return_all_detections,
        )

    def analyze(self):
        # Check that analyzer is LargeRecordingAnalyzer
        if not isinstance(self.analyzer, PerchAnalyzer):
            raise IncompatibleAnalyzerError(
                "LargeRecording can only be used with the Analyzer class"
            )

        # Set the file duration (does not read full audio into memory)
        # NOTE: This is the first opportunity for LR to read the file, so check for errors.
        try:
            self.duration = librosa.get_duration(filename=self.path)
        except audioread.exceptions.NoBackendError as e:
            print(e)
            raise AudioFormatError("Audio format could not be opened.")
        except FileNotFoundError as e:
            print(e)
            raise e
        except BaseException as e:
            print(e)
            raise AudioFormatError("Generic audio read error occurred from librosa.")

        # TODO: overlay is currently incompatible with LargeRecording. Implement this feature.

        # Analyze, though do not read the file all at once.
        self.analyzer.analyze_recording(self)
        self.analyzed = True


class PerchRemote(Remote):
    def __init__(
        self,
        api_endpoint="",
        api_key="",
        pid=None,
        processor_id=None,
        processor_type=None,
        aws_access_key_id="",
        aws_secret_access_key="",
        audio_directory=".",
        extraction_audio_directory=".",
        extraction_spectrogram_directory=".",
        analyzer=None,
        sleep_secs_on_empty_queue=3,
        runner_count=1,
        shutdown_on_empty_processing_queue=False,
        custom_param=None,  # Example custom parameter
    ):
        # Call the parent class's __init__ method
        super().__init__(
            api_endpoint=api_endpoint,
            api_key=api_key,
            pid=pid,
            processor_id=processor_id,
            processor_type=processor_type,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            audio_directory=audio_directory,
            extraction_audio_directory=extraction_audio_directory,
            extraction_spectrogram_directory=extraction_spectrogram_directory,
            analyzer=analyzer,
            sleep_secs_on_empty_queue=sleep_secs_on_empty_queue,
            runner_count=runner_count,
            shutdown_on_empty_processing_queue=shutdown_on_empty_processing_queue,
        )

    def _create_analyzer(self):
        print("_create_analyzer")
        analyzer = PerchAnalyzer()
        self.analyzer = analyzer

        # Store the Analyzer instance for later use.
        self._analyzers[self.analyzer_config_key] = analyzer
        self._analyzers_init_count = self._analyzers_init_count + 1

    def _analyze_file(self):
        data = self.queued_audio_dict

        analyzer_config = data["group"]["analyzer_config"]
        min_conf = analyzer_config.get("minimum_detection_confidence", None)
        self.min_conf_audio_extraction = analyzer_config.get(
            "minimum_detection_clip_confidence", 0.0
        )
        self.min_conf_spectrogram_extraction = analyzer_config.get(
            "minimum_detection_clip_confidence", 0.0
        )

        if self.analyzer_config_key not in self._analyzers:
            # Create analyzer if it doesn't already exist.
            self._create_analyzer()
        else:
            self.analyzer = self._analyzers[self.analyzer_config_key]

        if data["audio"].get("location", None):
            lat = data["audio"]["location"].get("latitude", None)
            lon = data["audio"]["location"].get("longitude", None)
        else:
            lat = None
            lon = None

        captured_local_date = data["audio"].get("captured_local_date", None)
        if lat and lon and captured_local_date:
            self.recording = PerchLargeRecording(
                self.analyzer,
                self.audio_filepath,
                min_conf=min_conf,
                lon=lon,
                lat=lat,
                date=datetime.strptime(captured_local_date, "%Y-%m-%d").date(),
                return_all_detections=True,
            )
        else:
            self.recording = PerchLargeRecording(
                self.analyzer,
                self.audio_filepath,
                min_conf=min_conf,
            )

        self.recording.analyze()
        pprint(self.recording.detections)

        self._set_checksum()


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
