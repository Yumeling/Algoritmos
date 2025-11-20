# modelo/arbol_decision.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class ArbolDecision:

    def __init__(self, criterio="entropia", max_depth=5, min_samples=2):
        self.criterio = criterio
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.tree = None

    # ===== IMPUREZA =====

    def entropia(self, y):
        valores, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return -np.sum(probs * np.log2(probs + 1e-9))

    def gini(self, y):
        valores, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return 1 - np.sum(probs ** 2)

    def impureza(self, y):
        return self.entropia(y) if self.criterio == "entropia" else self.gini(y)

    # ===== GANANCIA =====

    def ganancia(self, y, y_left, y_right):
        H_parent = self.impureza(y)
        w_left = len(y_left) / len(y)
        w_right = len(y_right) / len(y)
        return H_parent - (w_left * self.impureza(y_left) + w_right * self.impureza(y_right))

    # ===== NODOS =====

    def crear_nodo(self, X, y, depth):

        if depth >= self.max_depth or len(X) < self.min_samples or len(np.unique(y)) == 1:
            return {"leaf": True, "pred": self.clase_mayoritaria(y)}

        mejor_ganancia = 0
        mejor_col = None
        mejor_val = None
        mejor_splits = None

        for col in X.columns:
            valores = np.unique(X[col])
            for val in valores:

                left = X[col] <= val
                right = X[col] > val

                if left.sum() == 0 or right.sum() == 0:
                    continue

                y_left = y[left]
                y_right = y[right]
                g = self.ganancia(y, y_left, y_right)

                if g > mejor_ganancia:
                    mejor_ganancia = g
                    mejor_col = col
                    mejor_val = val
                    mejor_splits = (left, right)

        if mejor_col is None:
            return {"leaf": True, "pred": self.clase_mayoritaria(y)}

        left, right = mejor_splits

        return {
            "leaf": False,
            "col": mejor_col,
            "val": mejor_val,
            "left": self.crear_nodo(X[left], y[left], depth + 1),
            "right": self.crear_nodo(X[right], y[right], depth + 1),
        }

    def clase_mayoritaria(self, y):
        valores, counts = np.unique(y, return_counts=True)
        return valores[np.argmax(counts)]

    # ===== ENTRENAR =====

    def entrenar(self, X, y):
        y_array = np.array(y)
        self.tree = self.crear_nodo(X, y_array, depth=0)
        return self.tree

    # ===== PREDICCIÓN =====

    def predecir_fila(self, tree, fila):
        if tree["leaf"]:
            return tree["pred"]
        if fila[tree["col"]] <= tree["val"]:
            return self.predecir_fila(tree["left"], fila)
        else:
            return self.predecir_fila(tree["right"], fila)

    def predecir(self, X):
        return X.apply(lambda fila: self.predecir_fila(self.tree, fila), axis=1)

    # ===== ÁRBOL TEXTO =====

    def pretty_print(self, nodo=None, espacio=""):
        if nodo is None:
            nodo = self.tree

        if nodo["leaf"]:
            return espacio + "→ Predicción: " + str(nodo["pred"]) + "<br>"

        texto = (
            espacio
            + f"Si ({nodo['col']} <= {nodo['val']}):<br>"
            + self.pretty_print(nodo["left"], espacio + "&nbsp;&nbsp;&nbsp;")
            + espacio
            + f"Sino ({nodo['col']} > {nodo['val']}):<br>"
            + self.pretty_print(nodo["right"], espacio + "&nbsp;&nbsp;&nbsp;")
        )
        return texto

    # ===== ÁRBOL VISUAL (MATPLOTLIB, SIN GRAPHVIZ) =====

    def draw_tree(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis('off')

        def draw_node(node, x, y, dx):
            if node["leaf"]:
                ax.text(x, y, f"Leaf:\n{node['pred']}", ha='center',
                        bbox=dict(boxstyle="round", fc="lightgray"))
                return

            ax.text(x, y, f"{node['col']} <= {node['val']}", ha='center',
                    bbox=dict(boxstyle="round", fc="lightblue"))

            ax.plot([x, x - dx], [y - 0.2, y - 1], 'k-')
            draw_node(node["left"], x - dx, y - 1, dx / 2)

            ax.plot([x, x + dx], [y - 0.2, y - 1], 'k-')
            draw_node(node["right"], x + dx, y - 1, dx / 2)

        draw_node(self.tree, 0.5, 1.0, 0.25)
        fig.tight_layout()
        return fig
