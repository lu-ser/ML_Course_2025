import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from ipywidgets import interact, interactive, fixed, widgets
from IPython.display import display
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Button
from ipywidgets import interact, interactive, fixed, widgets, Layout, HBox, VBox
from IPython.display import display
from matplotlib.gridspec import GridSpec


def plot_regression_and_cost(x_train, y_train, compute_cost, w, b):
    """
    Crea una visualizzazione con due grafici affiancati:
    1. Grafico di regressione lineare con dati e predizioni
    2. Grafico del costo in funzione di w e b

    Parametri:
        x_train: np.array - dati di input
        y_train: np.array - valori target
        compute_cost: function - funzione per calcolare il costo
        w: float - peso attuale
        b: float - bias attuale
    """
    # Calcola il costo corrente
    current_cost = compute_cost(x_train, y_train, w, b)

    # Crea la figura con due subplot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        f"Minimizzazione Costo: Costo Corrente = {current_cost:.2f}", fontsize=16
    )

    # 1. Grafico della regressione lineare
    ax1.set_title("Regressione Lineare")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")

    # Disegna i punti di addestramento
    ax1.scatter(
        x_train, y_train, color="red", marker="x", s=100, label="Valore Attuale"
    )

    # Calcola i valori predetti
    x_line = np.linspace(min(x_train) - 0.5, max(x_train) + 0.5, 100)
    y_line = w * x_line + b
    ax1.plot(x_line, y_line, color="blue", label="Nostra Predizione")

    # Aggiungi i valori di costo per i punti
    for i in range(len(x_train)):
        pred_y = w * x_train[i] + b
        cost_point = (pred_y - y_train[i]) ** 2
        ax1.plot(
            [x_train[i], x_train[i]], [y_train[i], pred_y], "purple", linestyle="--"
        )
        ax1.text(
            x_train[i] + 0.05,
            (y_train[i] + pred_y) / 2,
            f"{cost_point:.0f}",
            color="purple",
        )

    ax1.text(
        x_line[-1] - 0.6,
        y_line[-1] - 30,
        f"cost = (1/m)∑({(pred_y - y_train[-1])**2:.0f} + ... ) = {current_cost:.2f}",
        color="purple",
    )

    ax1.legend()
    ax1.grid(True)

    # 2. Grafico del costo in funzione di w (con b corrente)
    ax2.set_title(f"Costo vs. w (b = {b})")
    ax2.set_xlabel("w")
    ax2.set_ylabel("Costo")

    # Genera valori di w e calcola il costo per ciascuno
    w_values = np.linspace(0, 200, 100)
    costs = [compute_cost(x_train, y_train, w_val, b) for w_val in w_values]

    # Disegna la curva del costo
    ax2.plot(w_values, costs, color="blue")

    # Evidenzia il punto corrente sulla curva del costo
    ax2.scatter(w, current_cost, color="red", s=100, label=f"costo a w={w}, b={b}")

    # Aggiungi una linea orizzontale al costo corrente
    ax2.axhline(y=current_cost, color="purple", linestyle="--")

    # Trova e segna il punto di costo minimo
    min_cost_idx = np.argmin(costs)
    min_w = w_values[min_cost_idx]
    min_cost = costs[min_cost_idx]

    # Mostra le coordinate del punto di costo minimo
    ax2.text(0, -max(costs) * 0.1, f"x={min_w:.3f} y={min_cost:.1f}", fontsize=10)

    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


def create_interactive_widget(
    x_train,
    y_train,
    compute_cost,
    w_initial=100,
    b_initial=100,
    w_min=0,
    w_max=200,
    w_step=1,
    b_min=0,
    b_max=200,
    b_step=1,
):
    """
    Crea un widget interattivo per esplorare i parametri della regressione

    Parametri:
        x_train: np.array - dati di input
        y_train: np.array - valori target
        compute_cost: function - funzione per calcolare il costo
        w_initial: float - valore iniziale per w
        b_initial: float - valore iniziale per b
        w_min: float - valore minimo per w
        w_max: float - valore massimo per w
        w_step: float - incremento per w
        b_min: float - valore minimo per b
        b_max: float - valore massimo per b
        b_step: float - incremento per b
    """
    interact(
        lambda w, b: plot_regression_and_cost(x_train, y_train, compute_cost, w, b),
        w=widgets.FloatSlider(
            value=w_initial, min=w_min, max=w_max, step=w_step, description="w1:"
        ),
        b=widgets.FloatSlider(
            value=b_initial, min=b_min, max=b_max, step=b_step, description="w0:"
        ),
    )


def plot_cost_surface(x_train, y_train, compute_cost):
    """
    Visualizza la superficie del costo in 3D al variare di w e b

    Parametri:
        x_train: np.array - dati di input
        y_train: np.array - valori target
        compute_cost: function - funzione per calcolare il costo
    """
    # Crea una griglia di valori w e b
    w_values = np.linspace(0, 200, 50)
    b_values = np.linspace(0, 200, 50)
    w_grid, b_grid = np.meshgrid(w_values, b_values)

    # Calcola il costo per ogni combinazione di w e b
    cost_grid = np.zeros_like(w_grid)
    for i in range(len(w_values)):
        for j in range(len(b_values)):
            cost_grid[j, i] = compute_cost(x_train, y_train, w_values[i], b_values[j])

    # Crea un grafico 3D
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Disegna la superficie del costo
    surface = ax.plot_surface(w_grid, b_grid, cost_grid, cmap="viridis", alpha=0.8)

    # Trova i parametri ottimali
    min_idx = np.unravel_index(np.argmin(cost_grid), cost_grid.shape)
    optimal_w = w_values[min_idx[1]]
    optimal_b = b_values[min_idx[0]]
    min_cost = cost_grid[min_idx]

    # Segna il punto di costo minimo
    ax.scatter([optimal_w], [optimal_b], [min_cost], color="red", s=100, label="Minimo")

    # Etichette e titolo
    ax.set_xlabel("w")
    ax.set_ylabel("b")
    ax.set_zlabel("Costo")
    ax.set_title("Superficie del costo al variare di w e b")
    fig.colorbar(surface, shrink=0.5, aspect=5)

    # Mostra i parametri ottimali
    print(f"Parametri ottimali: w = {optimal_w:.2f}, b = {optimal_b:.2f}")
    print(f"Costo minimo: {min_cost:.2f}")

    plt.legend()
    plt.tight_layout()
    plt.show()

    # Visualizza anche il grafico 2D con i parametri ottimali
    plot_regression_and_cost(x_train, y_train, compute_cost, optimal_w, optimal_b)


def find_optimal_parameters(x_train, y_train, compute_cost):
    """
    Calcola analiticamente i parametri ottimali della regressione lineare

    Parametri:
        x_train: np.array - dati di input
        y_train: np.array - valori target
        compute_cost: function - funzione per calcolare il costo
    """
    # Calcola i parametri ottimali usando il metodo dei minimi quadrati
    x_mean = np.mean(x_train)
    y_mean = np.mean(y_train)

    numerator = np.sum((x_train - x_mean) * (y_train - y_mean))
    denominator = np.sum((x_train - x_mean) ** 2)

    optimal_w = numerator / denominator
    optimal_b = y_mean - optimal_w * x_mean

    min_cost = compute_cost(x_train, y_train, optimal_w, optimal_b)

    print(f"Parametri ottimali: w = {optimal_w:.2f}, b = {optimal_b:.2f}")
    print(f"Costo minimo: {min_cost:.2f}")

    # Visualizza i risultati con i parametri ottimali
    plot_regression_and_cost(x_train, y_train, compute_cost, optimal_w, optimal_b)


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from ipywidgets import interact, interactive, fixed, widgets, Layout, HBox, VBox
from IPython.display import display, clear_output


def create_cost_level_visualizer(
    x_train, y_train, compute_cost, w_range=(-100, 500), b_range=(-200, 300)
):
    """
    Crea una visualizzazione con slider del costo che evidenzia la curva di livello corrispondente.
    Versione corretta che aggiorna correttamente la visualizzazione.

    Parametri:
        x_train: np.array - dati di input
        y_train: np.array - valori target
        compute_cost: function - funzione per calcolare il costo
        w_range: tuple - (min, max) per il range di w
        b_range: tuple - (min, max) per il range di b
    """
    # Crea la griglia di valori w e b
    w_values = np.linspace(w_range[0], w_range[1], 100)
    b_values = np.linspace(b_range[0], b_range[1], 100)
    w_grid, b_grid = np.meshgrid(w_values, b_values)

    # Calcola il costo per ogni combinazione di w e b
    cost_grid = np.zeros_like(w_grid)
    for i in range(len(w_values)):
        for j in range(len(b_values)):
            cost_grid[j, i] = compute_cost(x_train, y_train, w_values[i], b_values[j])

    # Trova il costo minimo e i relativi parametri
    min_idx = np.unravel_index(np.argmin(cost_grid), cost_grid.shape)
    optimal_w = w_values[min_idx[1]]
    optimal_b = b_values[min_idx[0]]
    min_cost = cost_grid[min_idx]

    # Calcola il range di costo per lo slider
    max_cost_display = np.percentile(
        cost_grid, 95
    )  # Limita per migliore visualizzazione

    # Definisci la funzione per generare il plot
    def create_plot(w_val, b_val, cost_level):
        """Funzione di rendering che crea tutti i grafici"""
        # Calcola il costo corrente
        current_cost = compute_cost(x_train, y_train, w_val, b_val)

        # Crea una nuova figura
        plt.close("all")  # Chiudi tutte le figure precedenti
        fig = plt.figure(figsize=(15, 10))

        # Layout della figura: due grafici in alto, uno sotto
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.5])

        # 1. Grafico di regressione
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.set_title("Housing Prices")
        ax1.set_xlabel("Dimensione (1000 meters squared)")
        ax1.set_ylabel("Prezzo (in 1000s of Euros)")

        # Disegna i punti di addestramento
        ax1.scatter(
            x_train, y_train, color="red", marker="x", s=100, label="Actual Value"
        )

        # Calcola i valori predetti
        x_line = np.linspace(min(x_train) - 0.5, max(x_train) + 0.5, 100)
        y_line = w_val * x_line + b_val
        ax1.plot(x_line, y_line, color="blue", label="Our Prediction")

        # Aggiungi i valori di costo per i punti
        cost_points = []
        for i in range(len(x_train)):
            pred_y = w_val * x_train[i] + b_val
            cost_point = (pred_y - y_train[i]) ** 2
            cost_points.append(cost_point)
            ax1.plot(
                [x_train[i], x_train[i]], [y_train[i], pred_y], "purple", linestyle="--"
            )
            ax1.text(
                x_train[i] + 0.05,
                min(y_train[i], pred_y) + abs(y_train[i] - pred_y) / 2,
                f" {cost_point:.0f}",
                color="purple",
            )

        # Formula di costo
        cost_formula = "cost = (1/m)*("
        for i, cost_point in enumerate(cost_points):
            if i > 0:
                cost_formula += " + "
            cost_formula += f"{cost_point:.0f}"
        cost_formula += f") = {current_cost:.0f}"

        y_pos = min(y_train) - (max(y_train) - min(y_train)) * 0.15
        ax1.text(min(x_train), y_pos, cost_formula, color="purple")

        ax1.legend()
        ax1.grid(True)

        # 2. Grafico delle curve di livello
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.set_title(f"Curve di livello - Evidenziata: {cost_level:.0f}")
        ax2.set_xlabel("w")
        ax2.set_ylabel("w0")

        # Genera livelli di costo per contour
        num_levels = 15
        contour_levels = np.geomspace(min_cost, max_cost_display, num_levels)

        # Disegna le curve di livello in grigio
        contour_bg = ax2.contour(
            w_grid, b_grid, cost_grid, levels=contour_levels, colors="gray", alpha=0.4
        )

        # Evidenzia la curva di livello selezionata
        highlighted_contour = ax2.contour(
            w_grid, b_grid, cost_grid, levels=[cost_level], colors=["red"], linewidths=2
        )
        ax2.clabel(highlighted_contour, inline=True, fontsize=10, fmt="%.0f")

        # Aggiungi un marker per la posizione corrente
        ax2.scatter([w_val], [b_val], color="blue", s=100)
        ax2.text(
            w_val + 5,
            b_val + 5,
            f"Cost attuale: {current_cost:.0f}",
            backgroundcolor="white",
            fontsize=9,
        )

        # 3. Grafico 3D della superficie del costo
        ax3 = fig.add_subplot(gs[1, :], projection="3d")
        ax3.set_title("E(y,w)")
        ax3.set_xlabel("w1")
        ax3.set_ylabel("w0")
        ax3.set_zlabel("E(y, w)")

        # Plot della superficie
        surf = ax3.plot_surface(
            w_grid,
            b_grid,
            np.minimum(cost_grid, max_cost_display),
            cmap="viridis",
            alpha=0.7,
            linewidth=0,
            antialiased=True,
        )

        # Aggiungi punti per la posizione corrente (w,b)
        ax3.scatter(
            [w_val],
            [b_val],
            [current_cost],
            color="blue",
            s=50,
            label="Parametri attuali",
        )

        # Trova punti sulla curva di livello evidenziata
        level_points = []
        step = 4  # Usa un passo per ridurre il numero di punti
        for i in range(0, len(w_values), step):
            for j in range(0, len(b_values), step):
                if (
                    abs(cost_grid[j, i] - cost_level) / cost_level < 0.05
                ):  # Tolleranza del 5%
                    level_points.append((w_values[i], b_values[j], cost_grid[j, i]))

        if level_points:
            level_w, level_b, level_z = zip(*level_points)
            ax3.scatter(
                level_w,
                level_b,
                level_z,
                color="red",
                s=20,
                alpha=0.7,
                label="Curva evidenziata",
            )

        ax3.legend()

        # Aggiungi un piano orizzontale al livello di costo evidenziato
        w_min, w_max = w_range
        b_min, b_max = b_range
        xx, yy = np.meshgrid([w_min, w_max], [b_min, b_max])
        zz = np.ones_like(xx) * cost_level
        ax3.plot_surface(xx, yy, zz, alpha=0.2, color="red")

        plt.tight_layout()
        return fig

    # Crea i widget
    w_slider = widgets.FloatSlider(
        value=optimal_w,
        min=w_range[0],
        max=w_range[1],
        step=(w_range[1] - w_range[0]) / 100,
        description="w1:",
        layout=Layout(width="500px"),
    )

    b_slider = widgets.FloatSlider(
        value=optimal_b,
        min=b_range[0],
        max=b_range[1],
        step=(b_range[1] - b_range[0]) / 100,
        description="w0:",
        layout=Layout(width="500px"),
    )

    # Crea uno slider per il livello di costo con una scala logaritmica
    cost_slider = widgets.FloatLogSlider(
        value=min_cost * 1.5,
        base=10,
        min=np.log10(max(min_cost, 1)),  # Evita logaritmi di numeri <= 0
        max=np.log10(max_cost_display),
        step=0.01,
        description="Errore:",
        layout=Layout(width="500px"),
        style={"description_width": "initial"},
    )

    # Etichetta per mostrare il costo attuale
    cost_label = widgets.Label(value=f"Errore attuale: {min_cost:.2f}")

    # Funzione per aggiornare la visualizzazione
    def update_plot(w, b, cost_level):
        with plot_output:
            clear_output(wait=True)
            fig = create_plot(w, b, cost_level)
            display(fig)
        cost_label.value = f"Errore attuale: {compute_cost(x_train, y_train, w, b):.2f}"

    # Funzione per tornare al minimo
    def go_to_minimum(b):
        w_slider.value = optimal_w
        b_slider.value = optimal_b
        cost_label.value = f"Errore attuale: {min_cost:.2f}"

    minimum_button = widgets.Button(
        description="Vai al minimo",
        button_style="info",
        tooltip="Imposta i parametri ai valori ottimali",
    )
    minimum_button.on_click(go_to_minimum)

    # Aggiungi un pulsante per selezionare il costo attuale
    def select_current_cost(b):
        current_cost = compute_cost(x_train, y_train, w_slider.value, b_slider.value)
        cost_slider.value = current_cost

    current_cost_button = widgets.Button(
        description="Seleziona errore attuale",
        button_style="warning",
        tooltip="Evidenzia la curva di livello per l'errore corrente",
    )
    current_cost_button.on_click(select_current_cost)

    # Layout dei widget
    controls = VBox(
        [
            HBox([w_slider, b_slider]),
            cost_label,
            cost_slider,
            HBox([minimum_button, current_cost_button]),
        ]
    )

    # Output per il plot
    plot_output = widgets.Output()

    # Mostra il layout
    display(controls)
    display(plot_output)

    # Collega gli slider all'aggiornamento
    out = widgets.interactive_output(
        update_plot, {"w1": w_slider, "w0": b_slider, "error_level": cost_slider}
    )

    # Inizializza il plot
    with plot_output:
        fig = create_plot(optimal_w, optimal_b, min_cost * 1.5)
        display(fig)

    return None


def plt_gradients(x, y, compute_cost, compute_gradient):
    """
    Visualizza il costo rispetto a 'w' e un grafico quiver che mostra i gradienti.

    Argomenti:
      x (ndarray): Dati di input
      y (ndarray): Valori target
      compute_cost: Funzione per calcolare il costo
      compute_gradient: Funzione per calcolare il gradiente
    """
    # Imposta b ad un valore fisso (come mostrato nel titolo della figura)
    b_fixed = 100

    # Crea una griglia di valori w per calcolare il costo
    w_array = np.linspace(0, 400, 200)
    cost = np.zeros_like(w_array)

    # Calcola il costo per ogni valore di w
    for i in range(len(w_array)):
        cost[i] = compute_cost(x, y, w_array[i], b_fixed)

    # Crea una figura con due sottografici
    plt.figure(figsize=(14, 6))
    gs = GridSpec(1, 2, width_ratios=[1, 1])

    # Primo grafico: Costo vs w
    ax1 = plt.subplot(gs[0])
    ax1.plot(w_array, cost, "b-")
    ax1.set_xlabel("w1")
    ax1.set_ylabel("Error")
    ax1.set_title("Error vs w1, with gradient; w0 set to 100")

    # Scegli alcuni punti specifici per evidenziare i gradienti
    w_points = [150, 220, 290]
    for w_val in w_points:
        # Calcola il costo e il gradiente per il punto specifico
        cost_val = compute_cost(x, y, w_val, b_fixed)
        dj_dw, _ = compute_gradient(x, y, w_val, b_fixed)

        # Segna il punto sul grafico
        ax1.plot(w_val, cost_val, "ro", markersize=8)

        # Aggiungi una linea tratteggiata arancione per indicare la tangente
        # Calcolo dei punti per la linea tangente
        w_range = 50
        x_tangent = np.array([w_val - w_range, w_val + w_range])
        y_tangent = cost_val + dj_dw * (x_tangent - w_val)
        ax1.plot(x_tangent, y_tangent, "r--")

        # Aggiungi l'annotazione con il valore del gradiente
        if abs(dj_dw) < 1:
            label = r"$\frac{\partial J}{\partial w} = 0$"
        else:
            label = r"$\frac{\partial J}{\partial w} = %d$" % dj_dw
        ax1.annotate(
            label,
            xy=(w_val, cost_val),
            xytext=(w_val + 10, cost_val - 3000),
            arrowprops=dict(arrowstyle="->"),
        )

    # Secondo grafico: Quiver plot dei gradienti
    ax2 = plt.subplot(gs[1])

    # Crea una griglia più densa di punti
    w_mesh, b_mesh = np.meshgrid(np.linspace(-100, 600, 20), np.linspace(-200, 200, 20))
    dj_dw_mesh = np.zeros_like(w_mesh)
    dj_db_mesh = np.zeros_like(b_mesh)

    # Calcola i gradienti per ogni punto della griglia
    for i in range(w_mesh.shape[0]):
        for j in range(w_mesh.shape[1]):
            dj_dw_mesh[i, j], dj_db_mesh[i, j] = compute_gradient(
                x, y, w_mesh[i, j], b_mesh[i, j]
            )

    # Importante: NON normalizzare i vettori, in modo che la lunghezza rappresenti la magnitudine
    # La scala è impostata in modo che le frecce siano visibili ma proporzionali alla magnitudine

    # Crea quiver plot - le frecce saranno più piccole vicino al minimo (circa w=200, b=100)
    quiver = ax2.quiver(
        w_mesh,
        b_mesh,
        dj_dw_mesh,
        dj_db_mesh,
        np.sqrt(dj_dw_mesh**2 + dj_db_mesh**2),  # Colore basato sulla magnitudine
        cmap="viridis",
        scale=2000,  # Fattore di scala per controllare la lunghezza delle frecce
        width=0.003,
    )  # Larghezza delle frecce

    ax2.set_xlabel("w1")
    ax2.set_ylabel("w0")
    ax2.set_title("Gradient shown in quiver plot")

    # Aggiungi un punto al minimo approssimativo
    ax2.plot(200, 100, "ro", markersize=6)

    plt.tight_layout()


def plt_divergence(p_hist, J_hist, x_train, y_train, compute_cost):
    """
    Funzione per visualizzare la divergenza dell'algoritmo di discesa del gradiente
    quando il learning rate è troppo grande.

    Args:
        p_hist (list): Storico dei parametri [w,b] per ogni iterazione
        J_hist (list): Storico dei valori di costo per ogni iterazione
        x_train (ndarray): Dati di training, feature x
        y_train (ndarray): Valori target di training
    """

    # Estrae i valori di w e b dallo storico dei parametri
    w_array = np.array([p[0] for p in p_hist])
    b_array = np.array([p[1] for p in p_hist])

    # Calcola il range dinamico basato sui valori massimi e minimi di w e b
    # con un margine aggiuntivo del 20%
    w_min, w_max = np.min(w_array), np.max(w_array)
    w_range = w_max - w_min
    w_min = w_min - 0.2 * w_range if w_range != 0 else w_min - 10000
    w_max = w_max + 0.2 * w_range if w_range != 0 else w_max + 10000

    b_min, b_max = np.min(b_array), np.max(b_array)
    b_range = b_max - b_min
    b_min = b_min - 0.2 * b_range if b_range != 0 else b_min - 10000
    b_max = b_max + 0.2 * b_range if b_range != 0 else b_max + 10000

    # Configura la figura con due subplot
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Cost escalates when learning rate is too large")

    # Crea una griglia di valori per w per visualizzare la curva della funzione di costo
    # nel grafico di sinistra (2D)
    w_space = np.linspace(w_min, w_max, 100)

    # Calcola il valore di costo per ogni w, usando il valore medio di b dai dati
    b_fixed = 100  # Questo è il valore mostrato nel titolo "Cost vs w, b set to 100"
    cost_vs_w = [compute_cost(x_train, y_train, w_i, b_fixed) for w_i in w_space]

    # Disegna la curva di costo vs w (la parabola di sfondo)
    ax[0].plot(w_space, cost_vs_w, "b-", linewidth=2)
    ax[0].set_title("Cost vs w1, w0 set to 100")
    ax[0].set_xlabel("w")
    ax[0].set_ylabel("Cost")

    # Evidenzia il percorso dell'algoritmo sul grafico 2D con una linea magenta
    ax[0].plot(w_array, J_hist, "magenta", linewidth=3)

    # Crea una griglia di valori per w e b per visualizzare la superficie della funzione di costo
    w_space_3d = np.linspace(w_min, w_max, 50)  # Ridotto a 50 punti per efficienza
    b_space_3d = np.linspace(b_min, b_max, 50)  # Ridotto a 50 punti per efficienza
    W, B = np.meshgrid(w_space_3d, b_space_3d)

    # Calcola il valore di costo per ogni combinazione di w e b
    Z = np.zeros_like(W)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            Z[i, j] = compute_cost(x_train, y_train, W[i, j], B[i, j])

    # Crea il grafico 3D della superficie della funzione di costo
    ax[1] = fig.add_subplot(1, 2, 2, projection="3d")
    ax[1].plot_surface(W, B, Z, alpha=0.3, cmap=cm.coolwarm)
    ax[1].set_xlabel("w")
    ax[1].set_ylabel("b")
    ax[1].set_zlabel("cost")
    ax[1].set_title("Cost vs (w0, w1)")

    # Traccia il percorso dell'algoritmo sulla superficie 3D
    ax[1].plot(w_array, b_array, J_hist, "magenta", linewidth=3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Imposta i limiti degli assi per una migliore visualizzazione
    max_cost = max(np.max(cost_vs_w), np.max(J_hist))
    ax[0].set_ylim(0, max_cost * 1.1)
