import pandas as pd
import numpy as np
from collections import Counter

class KModesFull:
   
    def __init__(self, df):
        self.df = df.copy()
        self.columns = df.columns.tolist()
        self.k = None
        self.centroids = None
        self.assignments = None

    
    def preprocess(self):
        self.df.replace("?", pd.NA, inplace=True)

        df_complete = self.df.dropna()
        self.k = df_complete.drop_duplicates().shape[0]

 
    def initialize_centroids(self):
        df_complete = self.df.dropna()
        unique_rows = df_complete.drop_duplicates()

        centroids = []
        for _, row in unique_rows.iterrows():
            centroids.append(list(row.values))

        self.centroids = centroids[:self.k]

 
    def hamming(self, row, centroid):
        dist = 0
        for v, c in zip(row, centroid):
            if pd.isna(v) or pd.isna(c):
                dist += 1
            elif v != c:
                dist += 1
        return dist

   
    def assign_clusters(self):
        assignments = []
        for _, row in self.df.iterrows():
            row_vals = list(row.values)

            best_idx = None
            best_dist = float("inf")

            for idx, centroid in enumerate(self.centroids):
                dist = self.hamming(row_vals, centroid)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx

            assignments.append(best_idx)

        self.assignments = assignments

   
    def update_centroids(self):
        new_centroids = []

        df_tmp = self.df.copy()
        df_tmp["Cluster"] = self.assignments

        for c in range(self.k):
            cluster_rows = df_tmp[df_tmp["Cluster"] == c]

            centroid = []
            for col in self.columns:
                moda = cluster_rows[col].mode(dropna=True)
                moda = moda.iloc[0] if not moda.empty else None
                centroid.append(moda)

            new_centroids.append(centroid)

        self.centroids = new_centroids

   
    def fit(self, max_iter=10):
        self.preprocess()
        self.initialize_centroids()

        for _ in range(max_iter):
            old_assignments = self.assignments.copy() if self.assignments else None

            self.assign_clusters()
            self.update_centroids()

            if self.assignments == old_assignments:
                break

        df_result = self.df.copy()
        df_result["Cluster"] = self.assignments

        for i, row in df_result.iterrows():
            centroid = self.centroids[row["Cluster"]]
            for j, col in enumerate(self.columns):
                if pd.isna(df_result.at[i, col]):
                    df_result.at[i, col] = centroid[j]

        return df_result
