import requests
from pprint import pprint
import boto3
import os
from botocore.exceptions import ClientError
from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer
import json
import hashlib
import time
from urllib.parse import urlparse


UNSPECIFIED = "Not specified"


class Remote:
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
    ):
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.pid = pid
        self.processor_id = processor_id
        self.processor_type = processor_type
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.queued_audio_dict = None
        self.audio_directory = audio_directory
        self.extraction_audio_directory = extraction_audio_directory
        self.extraction_spectrogram_directory = extraction_spectrogram_directory
        self.audio_file_obj = None
        self.audio_filepath = None
        self.analyzer = analyzer
        self.recording = None
        self._client = None
        self.detections = []
        self.file_checksum = None
        self.analyzer_duration_seconds = 0
        self.sleep_secs_on_empty_queue = sleep_secs_on_empty_queue
        self.min_conf_audio_extraction = 0.0
        self.min_conf_spectrogram_extraction = 0.0
        self.shutdown_on_empty_processing_queue = shutdown_on_empty_processing_queue
        self.runner_count = runner_count
        self._analyzers = {}
        self._analyzers_init_count = 0

    @property
    def api_headers(self):
        return {
            "BNL_APIKEY": self.api_key,
            "BNL_PROCESSOR_ID": self.processor_id,
        }

    def _return_queue_item(self):
        # TODO: Handle 404 and 500 with fibonacci backoff
        server_id = self.processor_id
        pid = self.pid
        data = {"server_id": server_id, "pid": pid}
        data["api_key"] = self.api_key  # Add api_key to outgoing request
        response = requests.post(
            f"{self.api_endpoint}/queues/audio/",
            json=data,
            headers=self.api_headers,
            verify=self.verify_request,
        )
        if response.status_code != 200:
            raise ConnectionError(
                f"Remote could not connect to API endpoint (status {response.status_code})."
            )
        data = response.json()
        if "id" in data:
            # Item returned, return this.
            return data
        if (
            data.get("safe_to_shutdown", False)
            and self.shutdown_on_empty_processing_queue
        ):
            # Shutdown here
            self._shutdown()
        return None

    def _save_results_to_server(self):
        # TODO: Handle 404 and 500 with fibonacci backoff
        data = self._format_results_for_api()
        data["api_key"] = self.api_key  # Add api_key to outgoing request
        audio_id = self.queued_audio_dict["id"]
        results_endpoint = f"{self.api_endpoint}/queues/audio/{audio_id}/results/"
        response = requests.post(
            results_endpoint,
            json=data,
            headers=self.api_headers,
            verify=self.verify_request,
        )
        if response.status_code != 201:
            raise ConnectionError(
                f"Remote could not connect to API endpoint (status {response.status_code})."
            )
        data = response.json()
        if data == {}:
            return None
        return data

    @property
    def instance_id(self):
        return self.processor_id if self.processor_id else UNSPECIFIED

    @property
    def instance_type(self):
        return self.processor_type if self.processor_type else UNSPECIFIED

    @property
    def verify_request(self):
        return not self.api_endpoint.startswith("http://")

    def _format_results_for_api(self):
        config_id = self.queued_audio_dict["group"]["analyzer_config"]["id"]
        data = {
            "detections": self.detections,
            "config_id": config_id,
            "duration_seconds": self.recording.duration,
            "analyzer_instance_id": self.instance_id,
            "analyzer_instance_type": self.instance_type,
            "analyzer_duration_seconds": self.analyzer_duration_seconds,
            "analyzer_version": self.analyzer.version,
            "file_checksum": self.file_checksum,
        }
        return data

    @property
    def client(self):
        if not self._client:
            self._client = boto3.client(
                "s3",
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key,
            )
        return self._client

    def _retrieve_file(self):
        # Get the file.
        data = self.queued_audio_dict["audio"]
        filename = os.path.basename(data["file_path"])
        self.audio_filepath = os.path.join(self.audio_directory, filename)
        bucket = data["file_source"]["s3_bucket"]
        object_key = data["file_path"]
        try:
            with open(self.audio_filepath, "wb") as f:
                self.client.download_fileobj(bucket, object_key, f)
        except ClientError as e:
            self.audio_file_obj = None
            self._cleanup_files()
            raise ConnectionError(
                f"Remote could not find audio file on S3 (error: {str(e)})."
            )

        self.audio_file_obj = f

    def _cleanup_files(self):
        os.remove(self.audio_filepath)
        detections = self.detections
        for detection in detections:
            if "extracted_audio_path" in detection:
                if os.path.exists(detection["extracted_audio_path"]):
                    os.remove(detection["extracted_audio_path"])
            if "extracted_spectrogram_path" in detection:
                if os.path.exists(detection["extracted_spectrogram_path"]):
                    os.remove(detection["extracted_spectrogram_path"])

    def _set_checksum(self):
        print("_set_checksum")
        self.file_checksum = hashlib.md5(
            open(self.audio_filepath, "rb").read()
        ).hexdigest()

    @property
    def analyzer_config_key(self):
        data = self.queued_audio_dict
        return hashlib.md5(
            json.dumps(data["group"]["analyzer_config"], sort_keys=True).encode("utf-8")
        ).hexdigest()

    def _create_analyzer(self):
        # Currently, only Birdnet-Analyzer is supported.
        # TODO: Add additional analyzers.

        data = self.queued_audio_dict
        analyzer_config = data["group"]["analyzer_config"]

        species_list = analyzer_config.get("species_list", [])

        analyzer_kwargs = {}

        if "base_version" in data["group"]["analyzer_config"]["analyzer"]:
            analyzer_kwargs["version"] = data["group"]["analyzer_config"]["analyzer"][
                "base_version"
            ]

        if len(species_list) != 0:
            analyzer_kwargs["custom_species_list"] = species_list

        # Handle custom models (which may be passed from the api)
        custom_model_file = analyzer_config["analyzer"].get("model_fp32_file", None)
        custom_labels_file = analyzer_config["analyzer"].get("labels_file", None)

        if custom_model_file:
            # Check to see if the file already exists.
            api_server_root = (
                urlparse(self.api_endpoint).scheme
                + "://"
                + urlparse(self.api_endpoint).hostname
            )

            model_filename = os.path.basename(custom_model_file)
            model_filepath = os.path.join(self.audio_directory, model_filename)

            if not os.path.exists(model_filepath):
                # Download the model file.
                url = f"{api_server_root}{custom_model_file}"
                r = requests.get(url)
                with open(model_filepath, "wb") as f:
                    f.write(r.content)

            labels_filename = os.path.basename(custom_labels_file)
            labels_filepath = os.path.join(self.audio_directory, labels_filename)

            if not os.path.exists(labels_filepath):
                # Download the labels file.
                url = f"{api_server_root}{custom_labels_file}"
                r = requests.get(url)
                with open(labels_filepath, "wb") as f:
                    f.write(r.content)

            analyzer_kwargs["classifier_model_path"] = model_filepath
            analyzer_kwargs["classifier_labels_path"] = labels_filepath

        analyzer = Analyzer(**analyzer_kwargs)
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

        if not self.analyzer_config_key in self._analyzers:
            # Create analyzer if it doesn't already exist.
            self._create_analyzer()
        else:
            self.analyzer = self._analyzers[self.analyzer_config_key]

        self.recording = Recording(
            self.analyzer,
            self.audio_filepath,
            min_conf=min_conf,
        )
        self.recording.analyze()
        pprint(self.recording.detections)

        self._set_checksum()

    def _extract_detections_as_audio(self):
        print("_extract_detections_as_audio")
        export_dir = self.extraction_audio_directory
        self.recording.extract_detections_as_audio(
            directory=export_dir, min_conf=self.min_conf_audio_extraction
        )

    def _extract_detections_as_spectrogram(self):
        print("_extract_detections_as_spectrogram")
        export_dir = self.extraction_spectrogram_directory
        self.recording.extract_detections_as_spectrogram(
            directory=export_dir, min_conf=self.min_conf_spectrogram_extraction
        )

    def _upload_extractions(self):
        # Audio and spectrograms.
        print("_upload_extractions")
        self.detections = self.recording.detections.copy()

        audio_bucket = self.queued_audio_dict["group"]["analyzer_config"][
            "extraction_audio_file_destination"
        ]["s3_bucket"]

        spectro_bucket = self.queued_audio_dict["group"]["analyzer_config"][
            "extraction_spectrogram_file_destination"
        ]["s3_bucket"]

        _uploaded_extractions = {}
        for detection in self.detections:
            source_file_path = self.queued_audio_dict["audio"]["file_path"]
            source_file_dir = os.path.dirname(source_file_path)
            if "extracted_audio_path" in detection:
                extract_file_name = os.path.basename(detection["extracted_audio_path"])
                key = f"{source_file_dir}/{extract_file_name}"
                success = self._upload_file_to_s3(
                    detection["extracted_audio_path"], audio_bucket, key
                )
                if success:
                    detection[
                        "extracted_audio_url"
                    ] = f"https://{audio_bucket}.s3.amazonaws.com/{key}"

            if "extracted_spectrogram_path" in detection:
                extract_file_name = os.path.basename(
                    detection["extracted_spectrogram_path"]
                )
                key = f"{source_file_dir}/{extract_file_name}"
                success = self._upload_file_to_s3(
                    detection["extracted_spectrogram_path"], spectro_bucket, key
                )
                if success:
                    detection[
                        "extracted_spectrogram_url"
                    ] = f"https://{spectro_bucket}.s3.amazonaws.com/{key}"

        self.uploaded_extractions = _uploaded_extractions

    def _upload_json(self):
        # Includes config (algo, min_conf, etc) and extractions
        data = self._format_results_for_api()
        analyzer_config = self.queued_audio_dict["group"]["analyzer_config"]
        data["analyzer_config"] = analyzer_config
        bucket = self.queued_audio_dict["group"]["analyzer_config"][
            "analysis_json_file_destination"
        ]["s3_bucket"]
        source_file_path = self.queued_audio_dict["audio"]["file_path"]
        key = f"{source_file_path}_data.json"
        body = json.dumps(data)
        self.client.put_object(Body=body, Bucket=bucket, Key=key)

    def _upload_file_to_s3(self, filepath, bucket, key):
        # Upload S3 file.
        # TODO: Change public-read to be configurable through the api.
        print("_upload_file_to_s3", key)
        try:
            self.client.upload_file(
                filepath, bucket, key, ExtraArgs={"ACL": "public-read"}
            )  # Returns no response. Will raise on error.
            return True
        except ClientError as e:
            print(e)
            return False

    def _shutdown(self):
        results_endpoint = f"{self.api_endpoint}/shutdown-instance/"
        data = {
            "analyzer_instance_id": self.instance_id,
            "number_of_runners": self.runner_count,
        }
        data["api_key"] = self.api_key  # Add api_key to outgoing request
        response = requests.post(
            results_endpoint,
            json=data,
            headers=self.api_headers,
            verify=self.verify_request,
        )
        print(response)
        os.system("sudo shutdown now -h")

    def process(self):
        # Retrieves item from queue, downloads, evaluates and returns as defined.
        # NOTE: Overly accepting try/except for catching and reporting all errors to api.
        # TODO: Breakout exceptions and provide more error handling options to api config.
        print("process")
        try:
            self.analyzer_duration_seconds = 0
            self.start_time = time.time()
            self.queued_audio_dict = self._return_queue_item()
            if self.queued_audio_dict:
                self._retrieve_file()
                self._analyze_file()
                self._extract_detections_as_audio()
                self._extract_detections_as_spectrogram()
                self._upload_extractions()
                self.analyzer_duration_seconds = round(time.time() - self.start_time, 2)
                # Processing complete, timer stopped.
                self._upload_json()
                self._cleanup_files()
                self._save_results_to_server()
        except BaseException as e:
            print(e)
            # TODO: Report back to the api.

    def run_queue(self):
        while True:
            self.process()
            if self.queued_audio_dict is None:
                print("queue empty, sleep")
                time.sleep(self.sleep_secs_on_empty_queue)
