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
        compute_cost: function - funzione per calcolare il s
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


def compute_cost(X, y, w, b):
    """
    Calcola la funzione di costo (errore quadratico medio)

    Args:
        X: Matrice delle caratteristiche
        y: Vettore dei valori target
        w: Vettore dei pesi
        b: Bias

    Returns:
        Valore della funzione di costo
    """
    m = X.shape[0]
    cost = 0

    # Calcolo della predizione
    y_pred = X @ w + b

    # Calcolo dell'errore quadratico medio
    cost = (1 / (2 * m)) * np.sum((y_pred - y) ** 2)

    return cost


def compute_gradient_matrix(X, y, w, b):
    """
    Calcola i gradienti della funzione di costo rispetto ai parametri w e b

    Args:
        X: Matrice delle caratteristiche
        y: Vettore dei valori target
        w: Vettore dei pesi
        b: Bias

    Returns:
        dj_dw: Gradiente rispetto ai pesi w
        dj_db: Gradiente rispetto al bias b
    """
    m, n = X.shape

    # Calcolo della predizione
    y_pred = X @ w + b

    # Calcolo dei gradienti
    dj_dw = (1 / m) * (X.T @ (y_pred - y))
    dj_db = (1 / m) * np.sum(y_pred - y)

    return dj_dw, dj_db


def gradient_descent(
    X, y, w_init, b_init, cost_function, gradient_function, alpha, num_iters
):
    """
    Implementazione dell'algoritmo del gradiente discendente

    Args:
        X: Matrice delle caratteristiche
        y: Vettore dei valori target
        w_init: Valori iniziali dei pesi
        b_init: Valore iniziale del bias
        cost_function: Funzione per calcolare il costo
        gradient_function: Funzione per calcolare i gradienti
        alpha: Tasso di apprendimento
        num_iters: Numero di iterazioni

    Returns:
        w: Pesi ottimizzati
        b: Bias ottimizzato
        J_history: Storia dei valori della funzione di costo
    """
    m = X.shape[0]
    w = w_init
    b = b_init
    J_history = []

    for i in range(num_iters):
        # Calcola gradienti
        dj_dw, dj_db = gradient_function(X, y, w, b)

        # Aggiorna parametri
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        # Salva il valore della funzione di costo
        if i % 100 == 0:
            J_history.append(cost_function(X, y, w, b))

    return w, b, J_history


def run_gradient_descent_lin(X, y, iterations=1000, alpha=1e-6):
    """
    Esegue l'algoritmo del gradiente discendente per trovare i parametri ottimali di un modello lineare.

    Args:
        X: Matrice delle caratteristiche di input
        y: Vettore dei valori target
        iterations: Numero di iterazioni per l'algoritmo del gradiente discendente
        alpha: Tasso di apprendimento (learning rate)

    Returns:
        w_out: Vettore dei pesi ottimizzati
        b_out: Valore del bias ottimizzato
    """
    m, n = X.shape  # m = numero di esempi, n = numero di caratteristiche

    # Inizializzazione dei parametri
    initial_w = np.zeros(n)  # Inizializza i pesi a zero
    initial_b = 0  # Inizializza il bias a zero

    # Esegui l'algoritmo del gradiente discendente
    w_out, b_out, hist_out = gradient_descent(
        X,
        y,
        initial_w,
        initial_b,
        compute_cost,
        compute_gradient_matrix,
        alpha,
        iterations,
    )

    print(
        f"Parametri trovati tramite gradiente discendente: w: {w_out}, b: {b_out:0.4f}"
    )

    return w_out, b_out


def plot_data(X, y, ax):
    pos = y[:, 0] == 1
    neg = y[:, 0] == 0
    ax.plot(X[pos, 0], X[pos, 1], "k+", label="Positive")
    ax.plot(X[neg, 0], X[neg, 1], "ko", label="Negative")
    ax.legend()


# Definizione dei colori
colors = {
    "blue": "#0096ff",
    "orange": "#FF9300",
    "darkred": "#C00000",
    "magenta": "#FF40FF",
    "purple": "#7030A0",
}

# Assegnazione dei colori a variabili
blue = colors["blue"]
orange = colors["orange"]
darkred = colors["darkred"]
magenta = colors["magenta"]
purple = colors["purple"]

# Lista dei colori
color_list = [blue, orange, darkred, magenta, purple]

# Importazione della libreria necessaria
from matplotlib.patches import FancyArrowPatch


def draw_threshold(ax, threshold):
    """Disegna una soglia sull'asse specificato."""
    y_limits = ax.get_ylim()
    x_limits = ax.get_xlim()

    # Riempimento delle aree sopra e sotto la soglia
    ax.fill_between(
        [x_limits[0], threshold], [y_limits[1], y_limits[1]], alpha=0.2, color=blue
    )
    ax.fill_between(
        [threshold, x_limits[1]], [y_limits[1], y_limits[1]], alpha=0.2, color=darkred
    )

    # Annotazioni per indicare le condizioni
    ax.annotate(
        "z >= 0",
        xy=[threshold, 0.5],
        xycoords="data",
        xytext=[30, 5],
        textcoords="offset points",
    )
    ax.annotate(
        "z < 0",
        xy=[threshold, 0.5],
        xycoords="data",
        xytext=[-50, 5],
        textcoords="offset points",
        ha="left",
    )

    # Freccia verso destra per z >= 0
    arrow_right = FancyArrowPatch(
        posA=(threshold, 0.5),
        posB=(threshold + 3, 0.5),
        color=darkred,
        arrowstyle="simple, head_width=5, head_length=10, tail_width=0.0",
    )
    ax.add_artist(arrow_right)

    # Freccia verso sinistra per z < 0
    arrow_left = FancyArrowPatch(
        posA=(threshold, 0.5),
        posB=(threshold - 3, 0.5),
        color=blue,
        arrowstyle="simple, head_width=5, head_length=10, tail_width=0.0",
    )
    ax.add_artist(arrow_left)


def sigmoid(z):
    """
    Funzione sigmoide

    Args:
        z: input scalare o numpy array

    Returns:
        g: sigmoide di z
    """
    return 1 / (1 + np.exp(-z))


def compute_cost_logistic_sq_err(X, y, w, b):
    """
    Calcola l'errore quadratico medio per la regressione logistica

    Args:
        X: dati di input (m, 1)
        y: target (m,)
        w: parametro del peso
        b: parametro del bias

    Returns:
        costo: errore quadratico medio
    """
    m = X.shape[0]

    # Calcola il valore z = w*x + b
    z = np.dot(X, w) + b

    # Applica la funzione sigmoide per ottenere le previsioni
    f_wb = sigmoid(z)

    # Calcola l'errore quadratico
    cost = np.sum((f_wb - y) ** 2) / (2 * m)

    return cost


def plt_logistic_squared_error(X, y):
    """
    Visualizza la superficie dell'errore quadratico per la regressione logistica

    Args:
        X: dati di input (m,)
        y: target (m,)
    """
    # Crea una griglia di valori per w e b
    wx, by = np.meshgrid(np.linspace(-6, 12, 50), np.linspace(10, -20, 40))
    points = np.c_[wx.ravel(), by.ravel()]
    cost = np.zeros(points.shape[0])

    # Calcola il costo per ogni combinazione di (w, b)
    for i in range(points.shape[0]):
        w, b = points[i]
        cost[i] = compute_cost_logistic_sq_err(X.reshape(-1, 1), y, w, b)

    # Riforma il costo nella stessa forma della griglia
    cost = cost.reshape(wx.shape)

    # Crea la figura 3D
    fig = plt.figure(figsize=(10, 8))
    fig.canvas.toolbar_visible = False
    fig.canvas.header_visible = False
    fig.canvas.footer_visible = False

    ax = fig.add_subplot(1, 1, 1, projection="3d")

    # Plotta la superficie del costo
    surf = ax.plot_surface(wx, by, cost, alpha=0.6, cmap=cm.jet)

    # Aggiungi etichette e titolo
    ax.set_xlabel("w", fontsize=16)
    ax.set_ylabel("b", fontsize=16)
    ax.set_zlabel("Cost", rotation=90, fontsize=16)
    ax.set_title('"Logistic" Squared Error Cost vs (w, b)')

    # Rendi trasparenti i pannelli degli assi
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    # Aggiungi una colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    plt.show()
    return fig


def plt_simple_example(x_train, y_train):
    """
    Visualizza un esempio di regressione logistica su dati di ammissione studenti.

    Args:
        x_train: array con il numero di domande corrette
        y_train: array con le etichette (0 per non ammesso, 1 per ammesso)
    """
    # Crea una nuova figura
    plt.figure(figsize=(8, 6))

    # Separa gli studenti ammessi e non ammessi
    non_ammessi = np.where(y_train == 0)[0]
    ammessi = np.where(y_train == 1)[0]

    # Plotta gli studenti non ammessi (cerchi blu)
    plt.scatter(
        x_train[non_ammessi],
        y_train[non_ammessi],
        color="blue",
        marker="o",
        s=100,
        alpha=0.8,
        edgecolor="black",
        linewidth=1,
        label="non ammesso",
    )

    # Plotta gli studenti ammessi (x rosse)
    plt.scatter(
        x_train[ammessi],
        y_train[ammessi],
        color="red",
        marker="x",
        s=100,
        alpha=0.8,
        linewidth=2,
        label="ammesso",
    )

    # Aggiungi titolo e etichette
    plt.title(
        "Esempio di Regressione Logistica sull'Ammissione degli Studenti", fontsize=14
    )
    plt.xlabel("Numero di Domande Corrette", fontsize=12)
    plt.ylabel("Ammissione (0/1)", fontsize=12)

    # Imposta i limiti degli assi
    plt.xlim(-0.5, max(x_train) + 0.5)
    plt.ylim(-0.1, 1.1)

    # Aggiungi legenda
    plt.legend(loc="upper left")

    # Mostra la griglia
    plt.grid(True, alpha=0.3)

    # Mostra il grafico
    plt.tight_layout()
    plt.show()

    return plt


def plot_two_logistic_loss_curves():
    # Creazione di predizioni da 0.001 a 0.999 (per evitare log(0))
    pred = np.linspace(0.001, 0.999, 1000)

    # Calcolo delle funzioni di perdita
    loss_y1 = -np.log(pred)  # loss quando t=1
    loss_y0 = -np.log(1 - pred)  # loss quando t=0

    # Creazione della figura con due subplot
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Plot per t=1
    ax[0].plot(pred, loss_y1, "b-", linewidth=2)
    ax[0].set_xlim([0, 1])
    ax[0].set_ylim([0, 5])
    ax[0].set_xlabel("$y(x_n,w)$")
    ax[0].set_ylabel("loss")
    ax[0].set_title("$t = 1$")

    # Annotazioni per t=1
    ax[0].annotate(
        "loss diminuisce quando la predizione\nsi avvicina dal target",
        xy=(0.3, 2),
        xytext=(0.2, 2.5),
        arrowprops=dict(facecolor="orange", shrink=0.05, width=2),
    )

    ax[0].annotate(
        "predizione\ncorrisponde\nal target",
        xy=(0.95, 0.1),
        xytext=(0.7, 0.8),
        arrowprops=dict(facecolor="orange", shrink=0.05, width=2),
    )

    # Plot per t=0
    ax[1].plot(pred, loss_y0, "b-", linewidth=2)
    ax[1].set_xlim([0, 1])
    ax[1].set_ylim([0, 5])
    ax[1].set_xlabel("$y(x_n,w)$")
    ax[1].set_title("$t = 0$")

    # Annotazioni per t=0
    ax[1].annotate(
        "loss aumenta quando la predizione\nsi allontana dal target",
        xy=(0.7, 2),
        xytext=(0.6, 2.5),
        arrowprops=dict(facecolor="orange", shrink=0.05, width=2),
    )

    ax[1].annotate(
        "predizione\ncorrisponde\nal target",
        xy=(0.05, 0.1),
        xytext=(0.25, 0.8),
        arrowprops=dict(facecolor="orange", shrink=0.05, width=2),
    )

    # Titolo generale
    plt.suptitle("Curve di Loss per Due Valori Target Categorici", fontsize=16)

    plt.tight_layout()
    plt.show()


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


def plot_logistic_cost_surface_3d(scale_type="linear"):
    """
    Visualizza la superficie di costo della regressione logistica in 3D

    Parametri:
    scale_type: 'linear' o 'log' per scala lineare o logaritmica
    """
    # Genera dati di esempio
    np.random.seed(42)
    n_samples = 100

    # Crea alcuni dati di esempio per una classificazione binaria
    X = np.random.randn(n_samples, 2)
    # Genera target basati su una linea di separazione
    t = (X[:, 0] + X[:, 1] > 0).astype(int)

    # Crea griglia per visualizzazione
    w0_vals = np.linspace(-5, 5, 50)
    w1_vals = np.linspace(-5, 5, 50)
    w0_grid, w1_grid = np.meshgrid(w0_vals, w1_vals)
    cost_grid = np.zeros_like(w0_grid)

    # Calcola il costo per ogni coppia di parametri (w0, w1)
    for i in range(len(w0_vals)):
        for j in range(len(w1_vals)):
            w0 = w0_vals[i]
            w1 = w1_vals[j]

            # Calcola le predizioni usando la sigmoide
            z = w0 + w1 * X[:, 0]  # Semplificato, uso solo la prima feature
            y_pred = 1 / (1 + np.exp(-z))

            # Calcola la funzione di perdita logistica
            epsilon = 1e-15  # Evita log(0)
            y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
            loss = -t * np.log(y_pred) - (1 - t) * np.log(1 - y_pred)
            cost = np.mean(loss)

            cost_grid[j, i] = cost

    # Applica scala logaritmica se richiesto
    if scale_type == "log":
        cost_grid = np.log(
            cost_grid + 1e-10
        )  # Aggiungi un piccolo valore per evitare log(0)
        title = "Superficie di Costo della Regressione Logistica (Scala Logaritmica)"
    else:
        title = "Superficie di Costo della Regressione Logistica (Scala Lineare)"

    # Crea la figura 3D
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Plot superficie
    surf = ax.plot_surface(
        w0_grid,
        w1_grid,
        cost_grid,
        cmap=cm.coolwarm,
        linewidth=0,
        antialiased=True,
        alpha=0.8,
    )

    # Etichette
    ax.set_xlabel("$w_0$")
    ax.set_ylabel("$w_1$")

    if scale_type == "log":
        ax.set_zlabel("$\log(E(y,w))$")
    else:
        ax.set_zlabel("$E(y,w)$")

    ax.set_title(title)

    # Aggiungi colorbar
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.tight_layout()
    plt.show()
