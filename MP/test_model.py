# import pickle
# import pandas as pd
# import sys


# def load_test_data():
#     try:
#         with open("../splitted_data/test_data.pkl", "rb") as f:
#             test_df = pickle.load(f)
#             if not isinstance(test_df, pd.DataFrame):
#                 raise TypeError("The file doesn't contains a pandas dataframe")
#     except TypeError as e:
#         print(e)
#         sys.exit(1)
#     except FileNotFoundError:
#         print("Error opening test data, Make sure you have splitted data.csv")
#         sys.exit(1)
#     return test_df

# if __name__ == "__main__":
#     test_df = load_test_data()
#     print(test_df.head(10))

import os
import platform
import subprocess
import webbrowser
import numpy as np
import plotly.graph_objects as go

# =====================================================================
# 📊 1. SIMULACIÓN DE DATOS (Tus 3 modelos)
# =====================================================================
epochs = np.arange(1, 51)

# Modelo 1: SGD (Convergencia lenta)
sgd_loss = np.exp(-epochs / 20) + 0.2
sgd_acc = 50 + 40 * (1 - np.exp(-epochs / 20))

# Modelo 2: Adam (Convergencia rápida y óptima)
adam_loss = np.exp(-epochs / 8) + 0.05
adam_acc = 50 + 48 * (1 - np.exp(-epochs / 8))


# =====================================================================
# 📐 2. CONSTRUCCIÓN DEL ESPACIO INTERACTIVO 3D
# =====================================================================
fig = go.Figure()

# Añadimos la trayectoria del Modelo 1 (SGD)
fig.add_trace(
    go.Scatter3d(
        x=epochs,
        y=sgd_loss,
        z=sgd_acc,
        mode="lines+markers",
        name="Model SGD",
        marker=dict(size=4),
        line=dict(color="red", width=3),
    )
)

# Añadimos la trayectoria del Modelo 2 (Adam)
fig.add_trace(
    go.Scatter3d(
        x=epochs,
        y=adam_loss,
        z=adam_acc,
        mode="lines+markers",
        name="Model Adam",
        marker=dict(size=4),
        line=dict(color="blue", width=3),
    )
)

# Configuración de los ejes del espacio 3D
fig.update_layout(
    title="Comparativa 3D Interactiva de Convergencia",
    scene=dict(
        xaxis_title="Épocas", yaxis_title="Loss", zaxis_title="Accuracy (%)"
    ),
    margin=dict(l=0, r=0, b=0, t=40),
)


# =====================================================================
# 💾 3. EXPORTAR A HTML Y DISPARAR VENTANA EMERGENTE COMPACTA
# =====================================================================
# Creamos un archivo HTML temporal con la gráfica
html_path = os.path.abspath("test_chart_3d.html")
fig.write_html(html_path)

url = f"file://{html_path}"
sistema = platform.system()

print("-> Abriendo visualización 3D en ventana flotante...")

try:
    if sistema == "Linux":
        # Ejecuta Google Chrome/Chromium en modo app independiente con tamaño fijo
        subprocess.Popen(
            ["google-chrome", f"--app={url}", "--window-size=850,550"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    elif sistema == "Darwin":  # macOS
        subprocess.Popen(
            [
                "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                f"--app={url}",
                "--window-size=850,550",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    elif sistema == "Windows":
        subprocess.Popen(
            ["cmd", "/c", "start", "chrome", f"--app={url}", "--window-size=850,550"],
            shell=True,
        )

except Exception:
    # Si por alguna configuración del entorno de la escuela no encuentra la ruta directa a Chrome,
    # usa el navegador por defecto del sistema para no romper la ejecución.
    webbrowser.open(url)