import pandas as pd
import numpy as np

class Normalizacion:
    def __init__(self, df, columnas):
        self.df = df.copy()
        self.columnas = columnas

    def minmax(self, nuevo_min, nuevo_max):
        df_norm = self.df.copy()

        for col in self.columnas:
            col_min = df_norm[col].min()
            col_max = df_norm[col].max()

            df_norm[col + "_minmax"] = (df_norm[col] - nuevo_min) / (nuevo_max - nuevo_min)

        return df_norm

    def zscore(self):
        df_norm = self.df.copy()

        for col in self.columnas:
            mean = df_norm[col].mean()
            std = df_norm[col].std(ddof=0)

            df_norm[col + "_zscore"] = (df_norm[col] - mean) / std

        return df_norm

    def log(self):
        df_norm = self.df.copy()

        for col in self.columnas:
            serie = pd.to_numeric(df_norm[col], errors="coerce")

            serie_log = serie.where(serie > 0)

            df_norm[col + "_log"] = np.log10(serie_log)

        return df_norm
