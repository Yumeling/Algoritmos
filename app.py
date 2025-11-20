from flask import Flask, render_template, request, redirect, url_for
import os
import pandas as pd
from utils.csv_analyzer import CSVAnalyzer
from modelo.normalizacion import Normalizacion
from modelo.kmedias import KMedias
from modelo.arbol_decision import ArbolDecision
from modelo.kmodas import KModesFull

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


global_analyzer1 = None
global_selected_columns1 = None
global_min_val = None
global_max_val = None

global_analyzer2 = None
global_df2 = None
columna_seleccionada = None
resultado_imputado = None
centroides_finales = None

global_analyzer3 = None
global_columns3 = None

global_analyzer5 = None
global_df5 = None
global_inputs5 = {}



@app.route("/")
def index():
    return render_template("index.html")

@app.route("/algoritmo1")
def algoritmo1():
    return render_template("algoritmo1_upload.html")

@app.route("/algoritmo2")
def algoritmo2():
    return render_template("algoritmo2_upload.html")

@app.route("/algoritmo3")
def algoritmo3():
    return render_template("algoritmo3_upload.html")

@app.route("/algoritmo1/upload", methods=["POST"])
def algoritmo1_upload():
    global global_analyzer1

    file = request.files.get("csv_file")
    if not file:
        return "No se subió archivo CSV"

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    global_analyzer1 = CSVAnalyzer(filepath)
    df = global_analyzer1.get_dataframe()
    columnas_num = global_analyzer1.numeric_columns()

    return render_template("algoritmo1_columnas.html", columnas=columnas_num)

@app.route("/algoritmo1/columnas", methods=["POST"])
def algoritmo1_columnas():
    global global_selected_columns1, global_min_val, global_max_val

    selected = request.form.getlist("columnas")
    global_selected_columns1 = selected
    global_min_val = float(request.form.get("min_val"))
    global_max_val = float(request.form.get("max_val"))

    return redirect(url_for("algoritmo1_resultado"))

@app.route("/algoritmo1/resultado")
def algoritmo1_resultado():
    global global_analyzer1, global_selected_columns1, global_min_val, global_max_val

    if global_analyzer1 is None or global_selected_columns1 is None:
        return "Error: Falta información previa."

    df_original = global_analyzer1.get_dataframe()
    normalizador = Normalizacion(df_original, global_selected_columns1)

    df_minmax = normalizador.minmax(global_min_val, global_max_val)
    df_zscore = normalizador.zscore()
    df_log = normalizador.log()

    return render_template(
        "algoritmo1_resultado.html",
        tabla_original=df_original.to_html(index=False),
        columnas=global_selected_columns1,
        tabla_minmax=df_minmax.to_html(index=False),
        tabla_zscore=df_zscore.to_html(index=False),
        tabla_log=df_log.to_html(index=False),
    )

@app.route("/algoritmo2/upload", methods=["POST"])
def algoritmo2_upload():
    global global_analyzer2, global_df2

    file = request.files.get("csv_file")
    if not file:
        return "No se subió archivo CSV"

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    global_analyzer2 = CSVAnalyzer(filepath)
    global_df2 = global_analyzer2.get_dataframe()

    columnas = global_df2.columns.tolist()

    return render_template("algoritmo2_columnas.html", columnas=columnas)


@app.route("/algoritmo2/columnas", methods=["POST"])
def algoritmo2_columnas():
    global columna_cluster, columna_feature

    columna_cluster = request.form.get("cluster_col")
    columna_feature = request.form.get("feature_col")

    return redirect(url_for("algoritmo2_resultado"))


@app.route('/algoritmo2/resultado')
def algoritmo2_resultado():
    global global_df2, columna_cluster, columna_feature

    if global_df2 is None or columna_cluster is None or columna_feature is None:
        return "Error: Falta información previa."
    df = global_df2.copy()
    km = KMedias(df, columna_feature)
    
    df_imputado, cluster_series, mapping, centroides = km.imputar_iterativo(columna_cluster)
    df_imputado['cluster_num'] = cluster_series

    return render_template(
        'algoritmo2_resultado.html',
        tabla_original=df.to_html(index=False),
        tabla_final=df_imputado.to_html(index=False)
    )

@app.route("/algoritmo3/upload", methods=["POST"])
def algoritmo3_upload():
    global global_analyzer3, global_columns3

    file = request.files.get("csv_file")
    if not file:
        return "No se subió archivo CSV"

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    global_analyzer3 = CSVAnalyzer(filepath)
    df = global_analyzer3.get_dataframe()
    global_columns3 = df.columns.tolist()

    return render_template("algoritmo3_columnas.html", columnas=global_columns3)


@app.route("/algoritmo3/columnas", methods=["POST"])
def algoritmo3_columnas():
    return redirect(url_for("algoritmo3_resultado"))


@app.route("/algoritmo3/resultado")
def algoritmo3_resultado():
    global global_analyzer3, global_columns3

    if global_analyzer3 is None or global_columns3 is None:
        return "Error: Falta información previa."

    df = global_analyzer3.get_dataframe()

    modelo = KModesFull(df)
    df_result = modelo.fit()

    centroides_dict = {idx: centroide for idx, centroide in enumerate(modelo.centroids)}

    df_clusters = df_result.copy()
    df_clusters = df_clusters[["Cluster"] + global_columns3]

    return render_template(
        "algoritmo3_resultado.html",
        tabla_original=df.to_html(index=False),
        tabla_predicha=df_result.to_html(index=False),
        tabla_clusters=df_clusters.to_html(index=False),
        centroides=centroides_dict,
        columnas=global_columns3
    )



from modelo.chimerge import ChiMerge

global_analyzer4 = None
global_cols4 = None

@app.route("/algoritmo4")
def algoritmo4():
    return render_template("algoritmo4_upload.html")


@app.route("/algoritmo4/upload", methods=["POST"])
def algoritmo4_upload():
    global global_analyzer4, global_cols4

    file = request.files.get("csv_file")
    if not file:
        return "No se subió archivo CSV"

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    global_analyzer4 = CSVAnalyzer(filepath)
    df = global_analyzer4.get_dataframe()
    global_cols4 = df.columns.tolist()

    return render_template("algoritmo4_columnas.html", columnas=global_cols4)


@app.route("/algoritmo4/columnas", methods=["POST"])
def algoritmo4_columnas():
    global atributo4, clase4

    atributo4 = request.form.get("atributo")
    clase4 = request.form.get("clase")

    return redirect(url_for("algoritmo4_resultado"))


@app.route("/algoritmo4/resultado")
def algoritmo4_resultado():
    global global_analyzer4, atributo4, clase4

    df = global_analyzer4.get_dataframe()

    modelo = ChiMerge(df, atributo4, clase4)
    df_result, intervalos, umbral = modelo.run()

    return render_template(
        "algoritmo4_resultado.html",
        tabla_original=df.to_html(index=False),
        tabla_resultado=df_result.to_html(index=False),
        intervalos=intervalos,
        atributo=atributo4,
        clase=clase4,
        umbral=umbral
    )


@app.route("/algoritmo5")
def algoritmo5():
    return render_template("algoritmo5_upload.html")


@app.route("/algoritmo5/upload", methods=["POST"])
def algoritmo5_upload():
    global global_analyzer5, global_df5

    file = request.files.get("csv_file")
    if not file:
        return "No se subió archivo CSV"

    path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(path)

    global_analyzer5 = CSVAnalyzer(path)
    global_df5 = global_analyzer5.get_dataframe()

    columnas = global_df5.columns.tolist()
    return render_template("algoritmo5_columnas.html", columnas=columnas)


@app.route("/algoritmo5/columnas", methods=["POST"])
def algoritmo5_columnas():
    global global_inputs5

    X_cols = request.form.getlist("columnas_x")
    y_col = request.form.get("columna_y")
    criterio = request.form.get("criterio")
    max_depth = int(request.form.get("max_depth"))
    min_samples = int(request.form.get("min_samples"))

    global_inputs5 = {
        "X_cols": X_cols,
        "y_col": y_col,
        "criterio": criterio,
        "max_depth": max_depth,
        "min_samples": min_samples,
    }

    return redirect(url_for("algoritmo5_resultado"))


@app.route("/algoritmo5/resultado")
def algoritmo5_resultado():
    global global_df5, global_inputs5

    if global_df5 is None or not global_inputs5:
        return "Error: faltan datos para el árbol."

    df = global_df5.copy()

    X_cols = global_inputs5["X_cols"]
    y_col = global_inputs5["y_col"]
    criterio = global_inputs5["criterio"]
    max_depth = global_inputs5["max_depth"]
    min_samples = global_inputs5["min_samples"]

    X = df[X_cols].copy()
    y = df[y_col].astype(str).str.strip()

    def map_si_no(serie):
        s = serie.astype(str).str.lower().str.strip()
        s = s.replace({"si": 1, "sí": 1, "sí": 1, "no": 0})
        return pd.to_numeric(s, errors="ignore")

    for col in X.columns:
        if X[col].dtype == object:
            X[col] = map_si_no(X[col])

    mask_unknown = y == "?"
    X_train = X[~mask_unknown]
    y_train = y[~mask_unknown]
    X_pred = X[mask_unknown]

    modelo = ArbolDecision(
        criterio=criterio,
        max_depth=max_depth,
        min_samples=min_samples
    )
    modelo.entrenar(X_train, y_train)

    arbol_texto = modelo.pretty_print()

    predicciones = []
    if not X_pred.empty:
        predicciones = modelo.predecir(X_pred).tolist()
        df.loc[mask_unknown, "Prediccion_Conclusion"] = predicciones

    import io
    import base64
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

    fig = modelo.draw_tree()
    pngImage = io.BytesIO()
    FigureCanvas(fig).print_png(pngImage)
    arbol_png = base64.b64encode(pngImage.getvalue()).decode('utf-8')

    return render_template(
        "algoritmo5_resultado.html",
        tabla_original=df.to_html(index=False),
        columnas_x=X_cols,
        columna_y=y_col,
        arbol_texto=arbol_texto,
        predicciones=predicciones,
        arbol_png=arbol_png
    )

if __name__ == "__main__":
    app.run(debug=True)
