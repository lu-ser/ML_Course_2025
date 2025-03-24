""" 
utilita_lab_comuni.py
    funzioni comuni a tutti i laboratori opzionali, Corso 1, Settimana 2 
"""

import numpy as np
import matplotlib.pyplot as plt

plt.style.use("./deeplearning.mplstyle")
colore_blu = "#0096ff"
colore_arancio = "#FF9300"
colore_rosso_scuro = "#C00000"
colore_magenta = "#FF40FF"
colore_viola = "#7030A0"
colori_dl = [
    colore_blu,
    colore_arancio,
    colore_rosso_scuro,
    colore_magenta,
    colore_viola,
]
dlc = dict(
    colore_blu="#0096ff",
    colore_arancio="#FF9300",
    colore_rosso_scuro="#C00000",
    colore_magenta="#FF40FF",
    colore_viola="#7030A0",
)


##########################################################
# Routine di Regressione
##########################################################


# Funzione per calcolare il costo
def calcola_costo_matrice(X, y, w, b, verbose=False):
    """
    Calcola il gradiente per la regressione lineare
     Args:
      X (ndarray (m,n)): Dati, m esempi con n caratteristiche
      y (ndarray (m,)) : valori target
      w (ndarray (n,)) : parametri del modello
      b (scalar)       : parametro del modello
      verbose : (Boolean) Se vero, stampa il valore intermedio f_wb
    Returns
      costo: (scalar)
    """
    m = X.shape[0]

    # calcola f_wb per tutti gli esempi.
    f_wb = X @ w + b
    # calcola costo
    costo_totale = (1 / (2 * m)) * np.sum((f_wb - y) ** 2)

    if verbose:
        print("f_wb:")
    if verbose:
        print(f_wb)

    return costo_totale


def calcola_gradiente_matrice(X, y, w, b):
    """
    Calcola il gradiente per la regressione lineare

    Args:
      X (ndarray (m,n)): Dati, m esempi con n caratteristiche
      y (ndarray (m,)) : valori target
      w (ndarray (n,)) : parametri del modello
      b (scalar)       : parametro del modello
    Returns
      dj_dw (ndarray (n,1)): Il gradiente del costo rispetto ai parametri w.
      dj_db (scalar):        Il gradiente del costo rispetto al parametro b.

    """
    m, n = X.shape
    f_wb = X @ w + b
    e = f_wb - y
    dj_dw = (1 / m) * (X.T @ e)
    dj_db = (1 / m) * np.sum(e)

    return dj_db, dj_dw


# Versione a ciclo del calcolo del costo multi-variabile
def calcola_costo(X, y, w, b):
    """
    calcola costo
    Args:
      X (ndarray (m,n)): Dati, m esempi con n caratteristiche
      y (ndarray (m,)) : valori target
      w (ndarray (n,)) : parametri del modello
      b (scalar)       : parametro del modello
    Returns
      costo (scalar)   : costo
    """
    m = X.shape[0]
    costo = 0.0
    for i in range(m):
        f_wb_i = np.dot(X[i], w) + b  # (n,)(n,)=scalar
        costo = costo + (f_wb_i - y[i]) ** 2
    costo = costo / (2 * m)
    return costo


def calcola_gradiente(X, y, w, b):
    """
    Calcola il gradiente per la regressione lineare
    Args:
      X (ndarray (m,n)): Dati, m esempi con n caratteristiche
      y (ndarray (m,)) : valori target
      w (ndarray (n,)) : parametri del modello
      b (scalar)       : parametro del modello
    Returns
      dj_dw (ndarray Shape (n,)): Il gradiente del costo rispetto ai parametri w.
      dj_db (scalar):             Il gradiente del costo rispetto al parametro b.
    """
    m, n = X.shape  # (numero di esempi, numero di caratteristiche)
    dj_dw = np.zeros((n,))
    dj_db = 0.0

    for i in range(m):
        err = (np.dot(X[i], w) + b) - y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err * X[i, j]
        dj_db = dj_db + err
    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_db, dj_dw
