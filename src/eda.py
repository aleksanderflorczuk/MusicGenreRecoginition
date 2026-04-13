import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


DATA_PATH = "../data/features_extended.csv"
OUTPUT_PATH = "../data/features_extended_scaled.csv"

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
plt.title("Number of tracks per genre")
plt.xlabel("Genre")
plt.ylabel("Count [-]")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

global_features = {
    "tempo": "Tempo [BPM]",
    "rms_mean": "RMS Energy [mean]",
    "rms_std": "RMS Energy [std]",
    "spectral_centroid_mean": "Spectral Centroid [Hz mean]",
    "spectral_centroid_std": "Spectral Centroid [Hz std]",
    "spectral_bandwidth_mean": "Spectral Bandwidth [Hz mean]",
    "spectral_bandwidth_std": "Spectral Bandwidth [Hz std]",
    "spectral_rolloff_mean": "Spectral Rolloff [Hz mean]",
    "spectral_rolloff_std": "Spectral Rolloff [Hz std]",
    "zcr_mean": "Zero Crossing Rate [mean]",
    "zcr_std": "Zero Crossing Rate [std]"
}

for col, label in global_features.items():
    plt.figure()
    sns.histplot(df[col], bins=30, kde=True)
    plt.title(f"Distribution of {label}")
    plt.xlabel(label)
    plt.ylabel("Count [-]")
    plt.tight_layout()
    plt.show()

mfcc_cols_mean = [f"mfcc_{i}_mean" for i in range(1, 14)]
mfcc_cols_std = [f"mfcc_{i}_std" for i in range(1, 14)]

plt.figure(figsize=(12, 5))
sns.boxplot(data=df[mfcc_cols_mean])
plt.title("Distribution of MFCC mean coefficients")
plt.xlabel("MFCC index")
plt.ylabel("Value [-]")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 5))
sns.boxplot(data=df[mfcc_cols_std])
plt.title("Distribution of MFCC std coefficients")
plt.xlabel("MFCC index")
plt.ylabel("Value [-]")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


chroma_mean_cols = [f"chroma_{i}_mean" for i in range(1, 13)]
chroma_std_cols = [f"chroma_{i}_std" for i in range(1, 13)]

plt.figure(figsize=(12, 5))
sns.boxplot(data=df[chroma_mean_cols])
plt.title("Distribution of Chroma mean")
plt.xlabel("Chroma index")
plt.ylabel("Value [-]")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 5))
sns.boxplot(data=df[chroma_std_cols])
plt.title("Distribution of Chroma std")
plt.xlabel("Chroma index")
plt.ylabel("Value [-]")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

corr = df[list(global_features.keys())].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap="coolwarm", center=0)
plt.title("Correlation of global audio features")
plt.tight_layout()
plt.show()

features_for_pca = mfcc_cols_mean + mfcc_cols_std + chroma_mean_cols + chroma_std_cols
scaled_features = StandardScaler().fit_transform(df[features_for_pca])

pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_features)

pca_df = pd.DataFrame(pca_result, columns=["PC1", "PC2"])
pca_df["genre"] = df["genre"]

plt.figure(figsize=(8, 6))
sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="genre", palette="tab10")
plt.title("PCA of MFCC + Chroma features")
plt.xlabel("Principal Component 1 [-]")
plt.ylabel("Principal Component 2 [-]")
plt.tight_layout()
plt.show()

scaler = StandardScaler()
scaled = scaler.fit_transform(df.drop(columns="genre"))

df_scaled = pd.DataFrame(scaled, columns=df.drop(columns="genre").columns)
df_scaled["genre"] = df["genre"]

df_scaled.to_csv(OUTPUT_PATH, index=False)
print(f"\n Scaled data saved to {OUTPUT_PATH}")



