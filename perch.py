import csv

import audioread
import librosa
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from birdnetlib import LargeRecording
from birdnetlib.analyzer import Detection
from birdnetlib.exceptions import (
    AudioFormatError,
    IncompatibleAnalyzerError,
)
from birdnetlib.utils import read_audio_segments


class PerchAnalyzer:
    def __init__(self):
        # Load the model using TFHub Lib.
        self.model = hub.load(
            "https://www.kaggle.com/models/google/bird-vocalization-classifier/TensorFlow2/bird-vocalization-classifier/8"
        )
        self.custom_species_list = []  # Not yet implemented.
        self.version = "Perch-8"
        self.labels = self.return_labels()
        self.species_dict = self.return_species_dict()

    def return_labels(self):
        # Define the file path
        file_path = "data/label.csv"

        # Open the file and read all lines except the first
        with open(file_path, "r") as file:
            lines = [line.strip() for i, line in enumerate(file) if i > 0]

        return lines

    def return_species_dict(self):
        file_path = "data/eBird_Taxonomy_v2021.csv"
        with open(file_path, mode="r", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            return {row["SPECIES_CODE"]: row for row in reader}

    def analyze_recording(self, recording):
        # print("analyze_recording, large mode", recording.filename)

        """
        Note: This differs from Birdnet-Analyzer in that it returns the top label
        from each segment, rather than any label above the threshold.
        """

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
            spec_code = self.labels[predicted_index]
            print("predicted ebird 2021 code", spec_code)
            sci_name = self.species_dict[spec_code]["SCI_NAME"]
            primary_com_name = self.species_dict[spec_code]["PRIMARY_COM_NAME"]
            label = f"{sci_name}_{primary_com_name}"
            print("label", label)
            print("Confidence in prediction:", confidence.numpy())

            # Store results
            if float(confidence) >= recording.minimum_confidence:
                results[str(start) + "-" + str(end)] = [(label, float(confidence))]

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
