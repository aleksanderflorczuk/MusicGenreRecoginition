import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


DATA_PATH = "../data/features.csv"
OUTPUT_PATH = "../data/features_scaled.csv"

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (9, 5)


df = pd.read_csv(DATA_PATH)

print("\n Dataset preview")
print(df.head())

print("\n Dataset info")
print(df.info())


print("\n Dataset shape:", df.shape)

print("\n Missing values")
print(df.isna().sum())


plt.figure()
sns.countplot(x="genre", data=df)
plt.title("Number of samples per genre")
plt.xlabel("Genre")
plt.ylabel("Number of tracks [-]")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# NUMERIC FEATURES
features = df.drop(columns="genre")

features = features.apply(pd.to_numeric, errors="coerce")
df[features.columns] = features


# GLOBAL AUDIO FEATURES
global_features_units = {
    "tempo": "Tempo [BPM]",
    "rms": "RMS Energy [-]",
    "spectral_centroid": "Spectral Centroid [Hz]",
    "spectral_bandwidth": "Spectral Bandwidth [Hz]",
    "spectral_rolloff": "Spectral Rolloff [Hz]",
    "zcr": "Zero Crossing Rate [-]"
}

for col, label in global_features_units.items():
    plt.figure()
    sns.histplot(df[col], bins=30, kde=True)
    plt.title(f"Distribution of {label}")
    plt.xlabel(label)
    plt.ylabel("Count [-]")
    plt.tight_layout()
    plt.show()


# FEATURES VS GENRE
compare_features_units = {
    "tempo": "Tempo [BPM]",
    "spectral_centroid": "Spectral Centroid [Hz]",
    "zcr": "Zero Crossing Rate [-]",
    "mfcc_1": "MFCC 1 [-]",
    "mfcc_2": "MFCC 2 [-]",
    "mfcc_3": "MFCC 3 [-]"
}

for col, label in compare_features_units.items():
    plt.figure()
    sns.boxplot(x="genre", y=col, data=df)
    plt.title(f"{label} across genres")
    plt.xlabel("Genre")
    plt.ylabel(label)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# MFCC AS A GROUP
mfcc_cols = [f"mfcc_{i}" for i in range(1, 14)]

plt.figure(figsize=(12, 5))
sns.boxplot(data=df[mfcc_cols])
plt.title("Distribution of MFCC coefficients")
plt.xlabel("MFCC coefficient index")
plt.ylabel("MFCC value [-]")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# CORRELATION (WITHOUT MFCC)
corr = df[list(global_features_units.keys())].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm", center=0)
plt.title("Correlation of global audio features")
plt.tight_layout()
plt.show()


# PCA ON MFCC
scaled_mfcc = StandardScaler().fit_transform(df[mfcc_cols])

pca = PCA(n_components=2)
mfcc_pca = pca.fit_transform(scaled_mfcc)

pca_df = pd.DataFrame(mfcc_pca, columns=["PC1", "PC2"])
pca_df["genre"] = df["genre"]

plt.figure(figsize=(8, 6))
sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="genre")
plt.title("PCA of MFCC features")
plt.xlabel("Principal Component 1 [-]")
plt.ylabel("Principal Component 2 [-]")
plt.tight_layout()
plt.show()


# FEATURE SCALING (FOR ML)
scaler = StandardScaler()
scaled = scaler.fit_transform(features)

df_scaled = pd.DataFrame(scaled, columns=features.columns)
df_scaled["genre"] = df["genre"]

df_scaled.to_csv(OUTPUT_PATH, index=False)
print(f"\n data saved to {OUTPUT_PATH}")

