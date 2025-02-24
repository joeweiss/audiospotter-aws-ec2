import os
import shutil
import time
import traceback
from pathlib import Path

import boto3
import numpy as np
import requests
import tensorflow as tf
from botocore.exceptions import ClientError
from chirp.inference import colab_utils
from etils import epath
from ml_collections import config_dict

colab_utils.initialize(use_tf_gpu=True, disable_warnings=True)

from chirp import audio_utils
from chirp.inference import embed_lib, tf_examples
from perch_hoplite.zoo import model_configs

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
        location_id=None,
        limit=None,
        destination_bucket="",
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
        self.location_id = location_id
        self.audio_list = []
        self.error_audio_list = []
        self.limit = limit
        self.embedding_filepath = f"{self.audio_directory}/perch"
        self.destination_bucket = destination_bucket
        self.dataset = None

    @property
    def api_headers(self):
        return {
            "BNL_APIKEY": self.api_key,
            "BNL_PROCESSOR_ID": self.processor_id,
        }

    def _return_audio_items(self):
        # TODO: Handle 404 and 500 with fibonacci backoff
        server_id = self.processor_id
        pid = self.pid
        data = {"server_id": server_id, "pid": pid}
        data["api_key"] = self.api_key  # Add api_key to outgoing request
        url = f"{self.api_endpoint}/queues/audio-list/"
        response = requests.post(
            url,
            json=data,
            headers=self.api_headers,
            verify=self.verify_request,
        )
        if response.status_code != 200:
            raise ConnectionError(
                f"Remote could not connect to API endpoint (status {response.status_code})."
            )
        data = response.json()
        # print(data)

        if "queued_audio" in data:
            # queued_audio returned, return this.
            self.dataset = data["dataset"]
            return data["queued_audio"]
        if data.get("safe_to_shutdown", False):
            if self.shutdown_on_empty_processing_queue:
                # Shutdown here
                self._shutdown()
            else:
                print(
                    "safe_to_shutdown is True, but shutdown_on_empty_processing_queue is False."
                )

        # print(data)
        # audio_list = data["queued_audio"]
        # self.dataset = data["dataset"]
        # # Confirm required file space (ideally this needs to be done prior to server launch)
        # total_bytes = sum([i["file_bytes"] for i in audio_list if i["file_bytes"]])
        # print("total GBs for audio: ", total_bytes / (1024**3))
        # total, used, free = shutil.disk_usage("/")
        # print("free space GBs: ", free / (1024**3))
        # print(total, used, free)

        # return audio_list

        return None

    def _save_results_to_server(self):
        # TODO: Handle 404 and 500 with fibonacci backoff
        data = {}
        data["api_key"] = self.api_key  # Add api_key to outgoing request
        data["queued_audio_ids"] = [i["id"] for i in self.audio_list]
        data["completed"] = True
        data["analyzer_instance_id"] = self.processor_id

        results_endpoint = f"{self.api_endpoint}/queues/audio-list/dataset/{self.dataset['id']}/results/"

        response = requests.post(
            results_endpoint,
            json=data,
            headers=self.api_headers,
            verify=self.verify_request,
        )
        if response.status_code != 200:
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

    @property
    def client(self):
        if not self._client:
            self._client = boto3.client(
                "s3",
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key,
            )
        return self._client

    def _retrieve_files(self):
        # Download the files to a directory with a progress bar
        for data in self.audio_list:
            print("-" * 80)
            print(data)

            audio = data["audio"]

            filename = os.path.basename(audio["file_path"])
            extension = os.path.splitext(filename)[1]
            audio_filepath = Path(self.audio_directory) / f"{audio['id']}{extension}"

            # Skip downloading if the file already exists
            if audio_filepath.exists():
                print("audio already exists")
                continue

            print(f"Downloading {filename}")
            bucket = audio["file_source"]["s3_bucket"]
            object_key = audio["file_path"]

            try:
                with open(audio_filepath, "wb") as f:
                    self.client.download_fileobj(bucket, object_key, f)
            except ClientError as e:
                self.error_audio_list.append({"data": audio, "e": str(e)})

    def _upload_embeddings(self):
        """Uploads all files from the given filepath to the specified S3 bucket.

        The directory structure in S3 will be:
        /location/{self.location_id}/<filename>
        """
        print("_upload_embeddings")
        s3_prefix = self.dataset["file_path"]
        bucket = self.dataset["file_destination"]["s3_bucket"]

        print(s3_prefix)
        print(bucket)

        ignore_files = {".DS_Store"}  # Use a set for faster lookups

        # Walk through all files and subdirectories in the embedding directory
        embedding_dir = f"{self.embedding_filepath}/embeddings"
        for root, _, files in os.walk(embedding_dir):
            print(files)
            for filename in files:
                if filename in ignore_files:  # Only check the filename, not full path
                    continue  # Skip ignored files

                local_path = os.path.join(root, filename)
                s3_key = os.path.join(
                    s3_prefix, os.path.relpath(local_path, embedding_dir)
                ).replace("\\", "/")

                # Upload file to S3
                self.client.upload_file(local_path, bucket, s3_key)
                print(f"Uploaded {local_path} to s3://{bucket}/{s3_key}")

    def _sync_embeddings_from_s3(self):
        """Deletes all local files in self.embedding_filepath and downloads everything from S3.

        Files are stored in:
        /location/{self.location_id}/ on S3 and are downloaded back to self.embedding_filepath.
        """
        s3_prefix = self.dataset["file_path"]
        bucket = self.dataset["file_destination"]["s3_bucket"]

        # Step 1: Delete all files in self.embedding_filepath
        if os.path.exists(self.embedding_filepath):
            shutil.rmtree(self.embedding_filepath)  # Remove everything
        os.makedirs(self.embedding_filepath, exist_ok=True)  # Recreate directory

        # Step 2: List all objects in S3 under the prefix
        response = self.client.list_objects_v2(Bucket=bucket, Prefix=s3_prefix)

        if "Contents" not in response:
            print("No files found in S3 under:", s3_prefix)
            return

        # Step 3: Download each file from S3
        for obj in response["Contents"]:
            s3_key = obj["Key"]
            relative_path = os.path.relpath(
                s3_key, s3_prefix
            )  # Remove prefix to get relative path
            local_path = os.path.join(self.embedding_filepath, relative_path)

            # Ensure the local directory exists
            os.makedirs(os.path.dirname(local_path), exist_ok=True)

            # Download file
            self.client.download_file(bucket, s3_key, local_path)
            print(f"Downloaded s3://{bucket}/{s3_key} to {local_path}")

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
        print(response.status_code)
        print("Shutting down now!!!!!!!!!")
        os.system("sudo shutdown now -h")

    def _cleanup_files(self):
        print("_cleanup_files")
        # Delete all files in self.audio_directory
        for file_name in os.listdir(self.audio_directory):
            file_path = os.path.join(self.audio_directory, file_name)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # Delete files and symbolic links
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # Delete directories

    def process(self):
        # Retrieves item from queue, downloads, evaluates and returns as defined.
        # NOTE: Overly accepting try/except for catching and reporting all errors to api.
        # TODO: Breakout exceptions and provide more error handling options to api config.
        print("process")

        try:
            self.analyzer_duration_seconds = 0
            self.start_time = time.time()
            self.audio_list = self._return_audio_items()
            if self.audio_list:
                self._retrieve_files()
                self._sync_embeddings_from_s3()
                self._extract_embeddings()
                self._upload_embeddings()
                self._save_results_to_server()
                self._cleanup_files()

        except BaseException as e:
            print(e)
            traceback.print_exc()
            # TODO: Report back to the api.

    def _extract_embeddings(self):
        # Run the Perch embeddings extraction as a test.

        """
        from embed_audio.ipynb
        """

        # @title Basic Configuration. { vertical-output: true }

        # @markdown Define the model: perch or birdnet are most common for birds.
        model_choice = "perch_8"  # @param['perch_8', 'humpback', 'multispecies_whale', 'surfperch', 'birdnet_V2.3']
        # @markdown Set the base directory for the project.
        # working_dir = '/tmp/agile'  #@param
        working_dir = self.embedding_filepath
        os.makedirs(working_dir, exist_ok=True)

        # Set the embedding and labeled data directories.
        embeddings_path = epath.Path(working_dir) / "embeddings"
        labeled_data_path = epath.Path(working_dir) / "labeled"
        embeddings_glob = embeddings_path / "embeddings-*"

        # OPTIONAL: Set up separation model.
        separation_model_key = "separator_model_tf"  # @param
        separation_model_path = ""  # @param

        config = config_dict.ConfigDict()
        config.embed_fn_config = config_dict.ConfigDict()
        config.embed_fn_config.model_config = config_dict.ConfigDict()

        # @markdown IMPORTANT: Select the target audio files.
        # @markdown source_file_patterns should contain a list of globs of audio files, like:
        # @markdown ['/home/me/*.wav', '/home/me/other/*.flac']
        # config.source_file_patterns = ['gs://chirp-public-bucket/soundscapes/powdermill/Recording*/*.wav']  #@param
        config.source_file_patterns = [
            f"{self.audio_directory}/*.wav",
            f"{self.audio_directory}/*.flac",
            f"{self.audio_directory}/*.WAV",
            f"{self.audio_directory}/*.FLAC",
        ]
        print(config.source_file_patterns)
        config.output_dir = embeddings_path.as_posix()

        preset_model_config = model_configs.get_preset_model_config(model_choice)

        # model_key, embedding_dim, model_config = model_configs.get_preset_model_config(
        #     model_choice)

        model_key = preset_model_config.model_key
        embedding_dim = preset_model_config.embedding_dim
        model_config = preset_model_config.model_config

        config.embed_fn_config.model_key = model_key
        config.embed_fn_config.model_config = model_config

        # Only write embeddings to reduce size.
        config.embed_fn_config.write_embeddings = True
        config.embed_fn_config.write_logits = False
        config.embed_fn_config.write_separated_audio = False
        config.embed_fn_config.write_raw_audio = False

        # @markdown File sharding automatically splits audio files into one-minute chunks
        # @markdown for embedding. This limits both system and GPU memory usage,
        # @markdown especially useful when working with long files (>1 hour).
        use_file_sharding = True  # @param {type:'boolean'}
        if use_file_sharding:
            config.shard_len_s = 60.0

        # Number of parent directories to include in the filename.
        config.embed_fn_config.file_id_depth = 1

        # Set up the embedding function, including loading models.
        embed_fn = embed_lib.EmbedFn(**config.embed_fn_config)
        print("\n\nLoading model(s)...")
        embed_fn.setup()

        # Create output directory and write the configuration.
        output_dir = epath.Path(config.output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        embed_lib.maybe_write_config(config, output_dir)

        # Create SourceInfos.
        source_infos = embed_lib.create_source_infos(
            config.source_file_patterns,
            num_shards_per_file=config.get("num_shards_per_file", -1),
            shard_len_s=config.get("shard_len_s", -1),
        )
        print(f"Found {len(source_infos)} source infos.")

        print("\n\nTest-run of model...")
        window_size_s = config.embed_fn_config.model_config.window_size_s
        sr = config.embed_fn_config.model_config.sample_rate
        z = np.zeros([int(sr * window_size_s)], dtype=np.float32)
        embed_fn.embedding_model.embed(z)
        print("Setup complete!")

        embed_fn.min_audio_s = 1.0
        record_file = (output_dir / "embeddings.tfrecord").as_posix()
        succ, fail = 0, 0

        existing_embedding_ids = embed_lib.get_existing_source_ids(
            output_dir, "embeddings-*"
        )

        new_source_infos = embed_lib.get_new_source_infos(
            source_infos, existing_embedding_ids, config.embed_fn_config.file_id_depth
        )

        print(
            f"Found {len(existing_embedding_ids)} existing embedding ids. \n"
            f"Processing {len(new_source_infos)} new source infos. "
        )

        try:
            audio_loader = lambda fp, offset: audio_utils.load_audio_window(
                fp,
                offset,
                sample_rate=config.embed_fn_config.model_config.sample_rate,
                window_size_s=config.get("shard_len_s", -1.0),
            )

            audio_iterator = audio_utils.multi_load_audio_window(
                filepaths=[s.filepath for s in new_source_infos],
                offsets=[s.shard_num * s.shard_len_s for s in new_source_infos],
                audio_loader=audio_loader,
            )

            with tf_examples.EmbeddingsTFRecordMultiWriter(
                output_dir=output_dir
            ) as file_writer:
                total_sources = len(new_source_infos)

                for idx, (source_info, audio) in enumerate(
                    zip(new_source_infos, audio_iterator), start=1
                ):
                    if not embed_fn.validate_audio(source_info, audio):
                        continue

                    file_id = source_info.file_id(config.embed_fn_config.file_id_depth)
                    offset_s = source_info.shard_num * source_info.shard_len_s
                    example = embed_fn.audio_to_example(file_id, offset_s, audio)

                    if example is None:
                        fail += 1
                        continue

                    file_writer.write(example.SerializeToString())
                    succ += 1

                    if (
                        idx % 10 == 0 or idx == total_sources
                    ):  # Log every 10 iterations and at the end
                        print(f"Processed {idx}/{total_sources} files...")

                file_writer.flush()

        finally:
            del audio_iterator
        print(f"\n\nSuccessfully processed {succ} source_infos, failed {fail} times.")

        fns = [fn for fn in output_dir.glob("embeddings-*")]
        ds = tf.data.TFRecordDataset(fns)
        parser = tf_examples.get_example_parser()
        ds = ds.map(parser)
        for ex in ds.as_numpy_iterator():
            print(ex["filename"])
            print(ex["timestamp_s"])
            print(ex["embedding"].shape, flush=True)
            break

    def run_queue(self):
        while True:
            self.process()
            if self.queued_audio_dict is None:
                print("queue empty, sleep")
                time.sleep(self.sleep_secs_on_empty_queue)
