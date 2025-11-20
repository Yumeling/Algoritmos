import pandas as pd
import numpy as np
from scipy.stats import chi2

class ChiMerge:
    """
    Implementación de Chi-Merge adaptada para el proyecto Flask.
    Usa df desde CSVAnalyzer y genera intervalos + etiqueta por intervalo.
    """

    def __init__(self, df, atributo, clase, umbral=None, alpha=0.05):
        self.df = df.copy()
        self.atributo = atributo
        self.clase = clase
        self.alpha = alpha
        self.umbral = umbral
        self.intervalos = None
        self.df_resultado = None

    # ---------------------------------------------------------------
    def run(self):
        datos = self.df[[self.atributo, self.clase]].dropna().copy()
        clases = sorted(datos[self.clase].unique())
        n_clases = len(clases)

        if n_clases == 0:
            raise ValueError("No existen clases para aplicar Chi-Merge.")

        # Si no se define umbral -> usar alfa
        if self.umbral is None:
            df_chi = n_clases - 1
            self.umbral = chi2.ppf(1 - self.alpha, df_chi)

        intervalos = self._crear_intervalos_iniciales(datos, clases)
        intervalos = self._fusionar_intervalos(intervalos)
        
        self.intervalos = intervalos

        # Asignar etiquetas al DF original
        self.df_resultado = self._asignar_etiquetas(self.df, intervalos)
        return self.df_resultado, intervalos, self.umbral

    # ---------------------------------------------------------------
    def _crear_intervalos_iniciales(self, datos, clases):
        agrupado = datos.groupby(self.atributo)[self.clase].value_counts().unstack(fill_value=0)
        agrupado = agrupado.reindex(columns=clases, fill_value=0).sort_index()

        intervalos = []
        for valor, fila in agrupado.iterrows():
            intervalos.append({
                "min": valor,
                "max": valor,
                "conteo": fila.values.astype(float)
            })

        return intervalos

    # ---------------------------------------------------------------
    def _fusionar_intervalos(self, intervalos):
        fusion = True

        while fusion and len(intervalos) > 1:
            fusion = False

            chis = []
            for i in range(len(intervalos) - 1):
                chi_val = self._calcular_chi2(intervalos[i], intervalos[i + 1])
                chis.append((chi_val, i))

            chis.sort(key=lambda x: x[0])
            chi_min, idx = chis[0]

            if chi_min < self.umbral:
                a = intervalos[idx]
                b = intervalos[idx + 1]

                nuevo = {
                    "min": a["min"],
                    "max": b["max"],
                    "conteo": a["conteo"] + b["conteo"]
                }

                intervalos[idx:idx + 2] = [nuevo]
                fusion = True

        return intervalos

    # ---------------------------------------------------------------
    def _calcular_chi2(self, a, b):
        try:
            obs = np.vstack([a["conteo"], b["conteo"]])
            fila_tot = obs.sum(axis=1, keepdims=True)
            col_tot = obs.sum(axis=0, keepdims=True)
            total = obs.sum()

            if total == 0:
                return 0.0

            esperado = fila_tot @ col_tot / total
            esperado[esperado == 0] = 1e-9

            return ((obs - esperado) ** 2 / esperado).sum()

        except:
            return 0.0

    # ---------------------------------------------------------------
    def _asignar_etiquetas(self, df, intervalos):
        etiquetas = []
        textos = []

        for _, fila in df.iterrows():
            val = fila[self.atributo]
            encontrado = False

            for i, intervalo in enumerate(intervalos):
                if intervalo["min"] <= val <= intervalo["max"]:
                    etiquetas.append(i)
                    textos.append(f"[{intervalo['min']}, {intervalo['max']}]")
                    encontrado = True
                    break

            if not encontrado:
                etiquetas.append(-1)
                textos.append("sin_intervalo")

        df_out = df.copy()
        df_out["chi_cluster"] = etiquetas
        df_out["intervalo"] = textos
        return df_out
