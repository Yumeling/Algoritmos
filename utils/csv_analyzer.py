import pandas as pd
import numpy as np
import io

class CSVAnalyzer:
    def __init__(self, ruta=None):
        self.df = None
        if ruta:
            self.load_from_path(ruta)

    def load_from_path(self, ruta, **read_kwargs):
        # Auto-detecta el separador usando Python engine
        self.df = pd.read_csv(ruta, sep=None, engine="python", **read_kwargs)
        self._convert_commas_to_dots()
        return self.df

    def load_from_fileobj(self, fileobj, filename=None, **read_kwargs):
        try:
            fileobj.stream.seek(0)
            content = fileobj.stream.read()
        except Exception:
            fileobj.seek(0)
            content = fileobj.read()

        if isinstance(content, bytes):
            content = content.decode('utf-8', errors='replace')

        self.df = pd.read_csv(io.StringIO(content), sep=None, engine="python", **read_kwargs)
        self._convert_commas_to_dots()
        return self.df

    def _convert_commas_to_dots(self):
        # Reemplaza comas por puntos y convierte a numérico donde sea posible
        for col in self.df.columns:
            if self.df[col].dtype == object:
                self.df[col] = self.df[col].str.replace(",", ".", regex=False)
                # Intenta convertir a numérico
                self.df[col] = pd.to_numeric(self.df[col], errors='ignore')

    def preview_html(self, n=10):
        if self.df is None:
            return "<p>No hay datos cargados</p>"
        return self.df.head(n).to_html(index=False)

    def detect_types(self):
        if self.df is None:
            return {}
        return self.df.dtypes.astype(str).to_dict()

    def numeric_columns(self):
        if self.df is None:
            return []
        return self.df.select_dtypes(include=['number']).columns.tolist()

    def normalize(self, method='minmax', columns=None):
        if self.df is None:
            raise ValueError("No hay dataframe cargado")
        cols = columns or self.numeric_columns()
        df = self.df.copy()
        for c in cols:
            serie = pd.to_numeric(df[c], errors='coerce')
            if method == 'minmax':
                mn, mx = serie.min(), serie.max()
                df[c + '_minmax'] = (serie - mn) / (mx - mn) if mx != mn else 0
            elif method == 'zscore':
                df[c + '_z'] = (serie - serie.mean()) / (serie.std(ddof=0) if serie.std(ddof=0)!=0 else 1)
            elif method == 'log':
                df[c + '_log'] = np.log(serie.clip(lower=1e-9) + 1)
            else:
                raise ValueError("Método desconocido")
        self.df = df
        return self.df

    def impute(self, method='mean', columns=None):
        if self.df is None:
            raise ValueError("No hay dataframe cargado")
        cols = columns or self.df.columns.tolist()
        df = self.df.copy()
        for c in cols:
            if method == 'mean':
                df[c] = pd.to_numeric(df[c], errors='coerce').fillna(pd.to_numeric(df[c], errors='coerce').mean())
            elif method == 'median':
                df[c] = pd.to_numeric(df[c], errors='coerce').fillna(pd.to_numeric(df[c], errors='coerce').median())
            elif method == 'mode':
                try:
                    v = df[c].mode().iloc[0]
                    df[c] = df[c].fillna(v)
                except Exception:
                    pass
            elif method == 'drop':
                df = df.dropna(subset=[c])
            elif method == 'interpolate':
                df[c] = pd.to_numeric(df[c], errors='coerce').interpolate().fillna(method='bfill').fillna(method='ffill')
            else:
                raise ValueError("Método de imputación desconocido")
        self.df = df
        return self.df

    def to_csv(self, path):
        if self.df is None:
            raise ValueError("No hay dataframe cargado")
        self.df.to_csv(path, index=False)
        return path

    def to_html(self):
        if self.df is None:
            return "<p>No hay datos</p>"
        return self.df.to_html(index=False)

    def get_dataframe(self):
        return self.df
