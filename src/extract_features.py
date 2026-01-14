import os
import librosa
import numpy as np
import pandas as pd

from config import AUDIO_DATASET_PATH
FEATURES = [
    "tempo",
    "zcr",
    "spectral_centroid",
    "spectral_bandwidth",
    "spectral_rolloff",
    "rms"] +[f"mfcc_{i}" for i in range(1,14)]

# tempo =bpm
# zcr zero crossing rate from
# rms energy how loud
# spectral centroid = srodek ciezkosci
# spectral bandwidth = jak bardzo czestotliwosci sa rozłozone wokol centroidu
# spectral rolloff czestotliwosci ponizej okreslonej enregii sygnalu
# MFCC = Mel frequency cepstral coefficients opisuja barwe dzwieku w sposob zblizony jak slyszy czlowiek opis charakteru brzmienia
# mfcc1 ogolna energia
# mfcc2-3 balans niskie/ wysopkie tony
# mfcc 4-6 barwa instrumentow
# mfcc 7-13 detale brzmienia
def extract_features(file_path):
    y,sr = librosa.load(file_path,sr=None)
    features = {}
    #BPM
    tempo = librosa.beat.tempo(y=y, sr=sr)
    features["tempo"] = float(tempo[0]) if len(tempo) > 0 else 0.0
    #TIME
    features["zcr"] = np.mean(librosa.feature.zero_crossing_rate(y))
    features["rms"] = np.mean(librosa.feature.rms(y=y))
    #SPECTRAL
    features["spectral_centroid"] = np.mean(
        librosa.feature.spectral_centroid(y=y, sr=sr)
    )
    features["spectral_bandwidth"] = np.mean(
        librosa.feature.spectral_bandwidth(y=y, sr=sr)
    )
    features["spectral_rolloff"] = np.mean(
        librosa.feature.spectral_rolloff(y=y, sr=sr)

    )
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(13):
        features[f"mfcc_{i + 1}"] = np.mean(mfccs[i])

    return features




def main():
    rows = []

    for genre in os.listdir(AUDIO_DATASET_PATH):
        genre_path = os.path.join(AUDIO_DATASET_PATH, genre)

        if not os.path.isdir(genre_path):
            continue

        print(f"processing genre: {genre}")

        for file in os.listdir(genre_path):
            if file.endswith(".wav"):
                file_path = os.path.join(genre_path, file)

                try:
                    feats = extract_features(file_path)
                    feats["genre"] = genre
                    rows.append(feats)
                except Exception as e:
                    print(f"error: {file_path}: {e}")

    df = pd.DataFrame(rows)
    df.to_csv("../data/features.csv", index=False)
    print("saved to: data/features.csv")


if __name__ == "__main__":
    main()




