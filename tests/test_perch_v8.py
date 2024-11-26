import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import librosa
import json

import pickle
import gzip


def return_labels():
    # Define the file path
    file_path = "tests/test_files/v8/label.csv"

    # Open the file and read all lines except the first
    with open(file_path, "r") as file:
        lines = [line.strip() for i, line in enumerate(file) if i > 0]

    return lines


def test_assert():
    assert 1 == 1


def test_silence():
    # Load the model.
    model = hub.load(
        "https://www.kaggle.com/models/google/bird-vocalization-classifier/TensorFlow2/bird-vocalization-classifier/8"
    )

    # Input: 5 seconds of silence as mono 32 kHz waveform samples.
    waveform = np.zeros(5 * 32000, dtype=np.float32)

    # Run the model, check the output.
    model_outputs = model.infer_tf(waveform[np.newaxis, :])

    # Examine various logits.
    print(model_outputs["label"].shape)
    print(model_outputs["genus"].shape)
    print(model_outputs["family"].shape)
    print(model_outputs["order"].shape)

    # Examine the embeddings.
    print(model_outputs["embedding"].shape)


def test_audio():
    # Load the model using TFHub Lib.
    model = hub.load(
        "https://www.kaggle.com/models/google/bird-vocalization-classifier/TensorFlow2/bird-vocalization-classifier/8"
    )

    path = "tests/test_files/soundscape.wav"

    # Expected Black-capped Chickadee
    waveform, rate = librosa.load(
        path, sr=32000, mono=True, res_type="kaiser_fast", offset=0.0, duration=5.0
    )

    # Run the model, check the output.
    model_outputs = model.infer_tf(waveform[np.newaxis, :])

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

    embeddings = model_outputs["embedding"]

    labels = return_labels()
    print(labels[predicted_index])
    print(embeddings)

    del waveform

    waveform, rate = librosa.load(
        path, sr=32000, mono=True, res_type="kaiser_fast", offset=0.0, duration=5.0
    )

    # Run the model, check the output.
    model_outputs = model.infer_tf(waveform[np.newaxis, :])

    # Convert logits to probabilities
    logits = model_outputs["label"]
    probabilities = tf.nn.softmax(logits, axis=-1)

    # Get the predicted class index
    predicted_index = np.argmax(probabilities)

    # Get the probability (confidence) of the predicted label
    confidence = probabilities[0, predicted_index]

    print("predicted_index", predicted_index)
    print("Confidence in prediction:", confidence.numpy())

    labels = return_labels()
    print(labels[predicted_index])

    # Get the top 5 class indices and their corresponding probabilities
    top_k = tf.math.top_k(probabilities, k=10)
    top_n_indices = top_k.indices.numpy().flatten()  # Ensure it's a flat NumPy array
    top_n_confidences = top_k.values.numpy().flatten()  # Ensure it's a flat NumPy array

    # Print the labels and probabilities
    for idx in range(len(top_n_indices)):
        predicted_index = top_n_indices[idx]  # Already a scalar after flattening
        print(labels[predicted_index], top_n_confidences[idx])

    embeddings = model_outputs["embedding"]

    print(embeddings)

    with open("single_query_v8.json", "w") as file:
        json.dump({"embedding_sample": embeddings.numpy().tolist()}, file, indent=4)

    data = []

    for i in range(0, 22):
        # Expected Blue Jay
        waveform, rate = librosa.load(
            path,
            sr=32000,
            mono=True,
            res_type="kaiser_fast",
            offset=float(i * 5),
            duration=5.0,
        )

        start_sec = float(i * 5)
        end_sec = start_sec + 5

        # Run the model, check the output.
        model_outputs = model.infer_tf(waveform[np.newaxis, :])

        # Convert logits to probabilities
        logits = model_outputs["label"]
        probabilities = tf.nn.softmax(logits, axis=-1)

        # Get the predicted class index
        predicted_index = np.argmax(probabilities)

        # Get the probability (confidence) of the predicted label
        confidence = probabilities[0, predicted_index]

        print("-" * 80)
        print("predicted_index", predicted_index)
        print("Confidence in prediction:", confidence.numpy())

        labels = return_labels()
        label = labels[predicted_index]
        confidence = confidence.numpy()
        print(label, confidence)

        embeddings = model_outputs["embedding"]
        data.append(
            {
                "label": label,
                "confidence": float(confidence),
                "filepath": path,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "embeddings_1280": [float(x) for x in embeddings.numpy().flatten()],
            }
        )

        del waveform
        del embeddings
        del model_outputs

    with open("output_v8.json", "w") as file:
        json.dump(data, file, indent=4)

    print("json")
    print(data[0]["label"])
    print(data[0]["confidence"])
    print(data[0]["embeddings_1280"][0])

    # Serialize each object into a binary string using Pickle
    serialized_data = np.array([pickle.dumps(obj) for obj in data], dtype=object)

    # Save the serialized data to a compressed file
    with gzip.open("compressed_data.npy.gz", "wb") as f:
        np.save(f, serialized_data)

    # Load the compressed data and deserialize it
    with gzip.open("compressed_data.npy.gz", "rb") as f:
        loaded_serialized_data = np.load(f, allow_pickle=True)

    # Deserialize the objects back to their original form
    loaded_data = np.array([pickle.loads(obj) for obj in loaded_serialized_data])

    # print(loaded_data)

    print("npy.gz")
    print(loaded_data[0]["label"])
    print(loaded_data[0]["confidence"])
    print(loaded_data[0]["embeddings_1280"][0])
