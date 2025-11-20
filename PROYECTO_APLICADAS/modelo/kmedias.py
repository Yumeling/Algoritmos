import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

class KMedias:
    def __init__(self, df, feature_col, max_iter=20, tol=1e-8, random_state=42):
        self.df = df.copy()
        self.feature_col = feature_col
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.random_state = int(random_state)
        self.df_result = None
        self.mapping_history = []
        self.centroides_finales = None
        self._original_missing_mask = None

    def _greedy_match(self, dist_mat):
        pairs = []
        n = dist_mat.shape[0]
        for i in range(n):
            for j in range(n):
                pairs.append((dist_mat[i, j], i, j))
        pairs.sort(key=lambda x: x[0])
        assigned_clusters = set()
        assigned_centroids = set()
        mapping = {}
        for d, i, j in pairs:
            if i in assigned_clusters or j in assigned_centroids:
                continue
            mapping[i] = j
            assigned_clusters.add(i)
            assigned_centroids.add(j)
            if len(assigned_clusters) == n:
                break
        return mapping

    def imputar_iterativo(self, cluster_col, verbose=False):
        df = self.df.copy()
        df[self.feature_col] = df[self.feature_col].replace("?", np.nan)
        df[self.feature_col] = pd.to_numeric(df[self.feature_col], errors="coerce")

        labels, uniques = pd.factorize(df[cluster_col])
        df['cluster_num'] = labels
        unique_labels = list(uniques)
        n_clusters = len(unique_labels)

        if n_clusters < 1:
            raise ValueError("No se detectaron clusters en la columna seleccionada.")

        self._original_missing_mask = df[self.feature_col].isna().copy()
        observed_mask = df[self.feature_col].notna()

        if observed_mask.sum() == 0:
            raise ValueError("La columna feature no tiene valores observados para estimar centroides.")

        global_mean = df.loc[observed_mask, self.feature_col].mean()
        df[self.feature_col] = df[self.feature_col].fillna(global_mean)

        prev_mapping = None
        final_mapping = None
        final_centroids = None

        for it in range(self.max_iter):
            try:
                if observed_mask.sum() >= n_clusters:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
                    kmeans.fit(df.loc[observed_mask, self.feature_col].values.reshape(-1, 1))
                else:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
                    kmeans.fit(df[self.feature_col].values.reshape(-1, 1))
                centroids = kmeans.cluster_centers_.flatten()
            except Exception:
                cluster_means_obs_series = df.loc[observed_mask].groupby(cluster_col)[self.feature_col].mean()
                cluster_means_obs = np.array([cluster_means_obs_series.get(lbl, np.nan) for lbl in unique_labels])
                cluster_means_obs = np.nan_to_num(cluster_means_obs, nan=global_mean)
                final_mapping = {i: i for i in range(n_clusters)}
                final_centroids = cluster_means_obs
                if verbose:
                    print(f"[iter {it}] fallback: usar medias observadas por label")
                break

            cluster_means_series_by_label = df.loc[observed_mask].groupby(cluster_col)[self.feature_col].mean()
            cluster_means = np.array([cluster_means_series_by_label.get(lbl, np.nan) for lbl in unique_labels])
            cluster_means = np.nan_to_num(cluster_means, nan=global_mean)

            centroids = np.array(centroids)
            if centroids.size != n_clusters:
                centroids = np.resize(centroids, n_clusters)
            dist_mat = np.abs(cluster_means.reshape(-1, 1) - centroids.reshape(1, -1))

            # Usar siempre greedy_match
            mapping = self._greedy_match(dist_mat)

            centroides_por_cluster = np.empty(n_clusters, dtype=float)
            for c in range(n_clusters):
                centroid_idx = mapping[c]
                centroides_por_cluster[c] = float(centroids[centroid_idx])

            for idx in df.index:
                if self._original_missing_mask.loc[idx]:
                    cnum = int(df.at[idx, 'cluster_num'])
                    df.at[idx, self.feature_col] = centroides_por_cluster[cnum]

            if prev_mapping is not None and prev_mapping == mapping:
                final_mapping = mapping
                final_centroids = centroides_por_cluster
                if verbose:
                    print(f"[iter {it}] convergió (mapping igual al previo).")
                break

            prev_mapping = mapping.copy()
            if it == self.max_iter - 1:
                final_mapping = mapping
                final_centroids = centroides_por_cluster
                if verbose:
                    print(f"[iter {it}] alcanzado max_iter; usando mapping actual.")

        self.df_result = df.copy()
        self.centroides_finales = final_centroids
        self.mapping_history.append(final_mapping)
        return df, df['cluster_num'], final_mapping, final_centroids
