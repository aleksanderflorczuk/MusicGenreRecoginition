import os
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

from config import AUDIO_DATASET_PATH

PROJECT_ROOT = Path(__file__).resolve().parents[1]

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


def main():
    rows = []

    for genre in os.listdir(AUDIO_DATASET_PATH):
        genre_path = os.path.join(AUDIO_DATASET_PATH, genre)
        if not os.path.isdir(genre_path):
            continue

        print(f"Processing genre: {genre}")

        for file in os.listdir(genre_path):
            if file.endswith(".wav"):
                file_path = os.path.join(genre_path, file)
                try:
                    feats = extract_features(file_path)
                    feats["genre"] = genre
                    rows.append(feats)
                except Exception as e:
                    print(f"Error {file_path}: {e}")

    df = pd.DataFrame(rows)
    df.to_csv(PROJECT_ROOT / "data" / "features_extended.csv", index=False)
    print("done!")

if __name__ == "__main__":
    main()
