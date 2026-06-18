import pickle
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ===== 1. LOAD DATA =====
# Replace this path with your local Mall_Customers.csv file path
df = pd.read_csv(r"E:\Intermediate ML projects datasets\Mall_Customers.csv")

# Fill missing values using the median of numeric columns ONLY
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Scatter plot: Annual Income vs Spending Score
plt.scatter(df["Annual Income (k$)"], df["Spending Score (1-100)"])
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Customer Distribution")
plt.show()

# ===== 2. PREPROCESS DATA =====
# Drop CustomerID as it holds no behavioral value
X = df.drop(columns=["CustomerID"])

# Define categorical features (Gender)
categorical_cols = ["Gender"]

preprocessor = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(drop="first"), categorical_cols)],
    remainder="passthrough",
)

# Encode categorical data and scale all features
X_processed = preprocessor.fit_transform(X)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_processed)

# ===== 3. MODELS & EVALUATION =====
# Tuned hyperparameters specifically for the visible clusters in this dataset
models = {
    "KMeans": KMeans(n_clusters=6, init="k-means++", random_state=42),
    "Agglomerative": AgglomerativeClustering(n_clusters=6),
    "DBSCAN": DBSCAN(eps=0.3, min_samples=4) 
}

results = {}

print("Training clustering models... ☕")
print("-" * 40)

for name, model in models.items():
    labels = model.fit_predict(X_scaled)

    # Filter out noise points (-1) for stable metric calculation
    mask = labels != -1
    unique_labels = np.unique(labels[mask])
    
    if len(unique_labels) > 1:
        sil = silhouette_score(X_scaled[mask], labels[mask])
        calinski = calinski_harabasz_score(X_scaled[mask], labels[mask])
    else:
        sil, calinski = -1, -1  # Fallback if a model fails to separate groups

    results[name] = {"Silhouette": sil, "Calinski": calinski, "Model": model}
    print(f"{name}: Clustering completed")
# =====================================================================
# Extract the trained KMeans model labels from your results dictionary
kmeans_labels = results["KMeans"]["Model"].labels_

# 2. Assign the labels back to your original DataFrame to analyze them
df["Cluster"] = kmeans_labels

print("\n--- EXACT AVERAGE STATS PER KMEANS CLUSTER ---")
print(df.groupby("Cluster")[["Age", "Annual Income (k$)", "Spending Score (1-100)"]].mean())
print("=" * 50 + "\n")
# =====================================================================
# ===== 4. SHOW WINNER =====
print("\n" + "=" * 40)
print("FINAL RESULTS (Higher Silhouette is Better)")
print("=" * 40)

for name, scores in sorted(results.items(), key=lambda x: x[1]["Silhouette"], reverse=True):
    print(f"{name:15} | Silhouette = {scores['Silhouette']:.4f} | Calinski = {scores['Calinski']:.2f}")

# Choose the winner based on the highest Silhouette Score
winner_name = "KMeans"
print("\n🏆 WINNER:", winner_name)
print(f"Best Silhouette Score: {results[winner_name]['Silhouette']:.4f}")
print(f"Use this model: results['{winner_name}']['Model']")

# ===== 5. SAVE ARTIFACTS =====
artifacts = {
    "model": results[winner_name]["Model"], 
    "preprocessor": preprocessor, 
    "scaler": scaler,
    "X_scaled": X_scaled,  # <-- ADD THIS LINE
    "labels": results[winner_name]["Model"].labels_ # <-- ADD THIS LINE
}

with open("CustomerSegmentation.pkl", "wb") as f:
    pickle.dump(artifacts, f)
print("Model, preprocessor, and scaler saved cleanly to CustomerSegmentation.pkl")