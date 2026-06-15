import argparse
import os

import librosa
import numpy as np
import pandas as pd

from project_config import (
    add_config_argument,
    load_config,
    project_path,
    write_experiment_manifest,
)

FEATURES = ([
    "tempo",
    "zcr_mean", "zcr_std",
    "rms_mean", "rms_std",
    "spectral_centroid_mean", "spectral_centroid_std",
    "spectral_bandwidth_mean", "spectral_bandwidth_std",
    "spectral_rolloff_mean", "spectral_rolloff_std"
] + [f"mfcc_{i}_mean" for i in range(1, 14)] + [f"mfcc_{i}_std" for i in range(1, 14)]+
    [f"chroma_{i}_mean" for i in range(1,13)] + [f"chroma_{i}_std" for i in range(1,13)])

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    features = {}

    tempo = librosa.beat.tempo(y=y, sr=sr)
    features["tempo"] = float(tempo[0]) if len(tempo) > 0 else 0.0

    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)

    features["zcr_mean"] = np.mean(zcr)
    features["zcr_std"] = np.std(zcr)

    features["rms_mean"] = np.mean(rms)
    features["rms_std"] = np.std(rms)

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)

    features["spectral_centroid_mean"] = np.mean(centroid)
    features["spectral_centroid_std"] = np.std(centroid)

    features["spectral_bandwidth_mean"] = np.mean(bandwidth)
    features["spectral_bandwidth_std"] = np.std(bandwidth)

    features["spectral_rolloff_mean"] = np.mean(rolloff)
    features["spectral_rolloff_std"] = np.std(rolloff)


    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(13):
        features[f"mfcc_{i+1}_mean"] = np.mean(mfccs[i])
        features[f"mfcc_{i+1}_std"] = np.std(mfccs[i])

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    for i in range(12):
         features[f"chroma_{i+1}_mean"] = np.mean(chroma[i])
         features[f"chroma_{i+1}_std"] = np.std(chroma[i])

    return features


def parse_args():
    parser = argparse.ArgumentParser(description="Extract audio features.")
    add_config_argument(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    environment_variable = config["audio_dataset_environment_variable"]
    audio_dataset_path = os.getenv(environment_variable)
    if not audio_dataset_path:
        raise EnvironmentError(
            f"Set the {environment_variable} environment variable to the GTZAN directory."
        )

    rows = []

    for genre in sorted(os.listdir(audio_dataset_path)):
        genre_path = os.path.join(audio_dataset_path, genre)
        if not os.path.isdir(genre_path):
            continue

        print(f"Processing genre: {genre}")

        for file in sorted(os.listdir(genre_path)):
            if file.lower().endswith(".wav"):
                file_path = os.path.join(genre_path, file)
                try:
                    feats = extract_features(file_path)
                    feats["genre"] = genre
                    rows.append(feats)
                except Exception as e:
                    print(f"Error {file_path}: {e}")

    if not rows:
        raise RuntimeError("No audio features were extracted; the existing dataset was not changed.")

    df = pd.DataFrame(rows)
    missing_columns = [column for column in FEATURES if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Extracted dataset is missing features: {missing_columns}")

    output_path = project_path(config["feature_file"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".tmp")
    df.to_csv(temporary_path, index=False)
    temporary_path.replace(output_path)
    print(f"Saved {len(df)} rows to: {output_path}")
    print(f"Updated manifest: {write_experiment_manifest(config, 'feature_extraction')}")

if __name__ == "__main__":
    main()
