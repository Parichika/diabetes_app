#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  9 15:58:37 2025

@author: parichikaphumikakrak
"""

import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns

# Set Streamlit page configuration
st.set_page_config(page_title="K-Means Iris Clustering", layout="wide")

# Page title with emoji
st.title("🔍 K-Means Clustering App with Iris Dataset")
st.caption("Created by **Parichika Phumikakrak**")

# Load the Iris dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
feature_names = iris.feature_names

# Sidebar controls
with st.sidebar:
    st.header("🧪 Clustering Configuration")
    k = st.slider("🔢 Select number of clusters (k)", 2, 10, 3)
    show_data = st.checkbox("📊 Show Raw Dataset", value=False)

# Optional: Show dataset
if show_data:
    with st.expander("📄 Preview Iris Dataset"):
        st.dataframe(X)

# KMeans clustering
kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
kmeans.fit(X)
labels = kmeans.labels_

# PCA for 2D projection
pca = PCA(n_components=2)
reduced = pca.fit_transform(X)
reduced_df = pd.DataFrame(reduced, columns=["PCA1", "PCA2"])
reduced_df["Cluster"] = labels

# Color palette
palette = sns.color_palette("hsv", k)
colors = [palette[i] for i in reduced_df["Cluster"]]

# Plot the clusters
fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(reduced_df["PCA1"], reduced_df["PCA2"], c=colors, s=60, edgecolors='k', alpha=0.8)
ax.set_title("🌸 Clusters (PCA Projection)", fontsize=14)
ax.set_xlabel("PCA Component 1")
ax.set_ylabel("PCA Component 2")

# Add legend
for cluster in range(k):
    ax.scatter([], [], c=[palette[cluster]], label=f"Cluster {cluster}", edgecolors='k')
ax.legend(title="Clusters")

# Show plot
st.pyplot(fig)

# Summary message
st.success(f"✅ Successfully clustered {len(X)} data points into **{k} clusters**!")

# Bonus info
with st.expander("ℹ️ About this App"):
    st.markdown("""
    - This app uses **K-Means clustering** with **PCA** for visualization.
    - The dataset used is the classic **Iris dataset** from scikit-learn.
    - Adjust the number of clusters in the sidebar to explore different segmentations.
    """)