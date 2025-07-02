import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from lap import lapjv
import time
from scipy.spatial.distance import cdist
from concurrent.futures import ThreadPoolExecutor
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerPatch
from matplotlib.path import Path
from matplotlib.gridspec import GridSpec
from slmsuite.holography.algorithms import Hologram
import math

# Parámetros
ATOM_DIAMETER = 1.0  # Diámetro del átomo en micrómetros
GRID_SPACING = 5.0   # Separación entre puntos del grid en micrómetros
BLOCKS = 2  # Número de bloques en cada dirección
GRID_SIZE = 40 # Tamaño del grid (número de puntos)
OCCUPANCY = 0.65 # Porcentaje de ocupación del array inicial
TARGET_SIZE = 25 # Tamaño del array objetivo (número de puntos)
ALPHA = 2 # Parámetro de la función de costo
STEPS = 20  # Número de interpolaciones
SYSTEM_SIZE_UM = GRID_SIZE * GRID_SPACING  # Tamaño total del sistema en micrómetros

# Parámetros del holograma
HOLOGRAM_SIZE = (256, 256)  # Tamaño del holograma
HOLOGRAM_ITERATIONS = 20  # Iteraciones para el algoritmo GS
HOLOGRAM_PHYSICAL_SIZE_UM = GRID_SIZE * GRID_SPACING  # Tamaño físico del holograma en micrómetros
HOLOGRAM_PIXEL_SIZE_UM = HOLOGRAM_PHYSICAL_SIZE_UM / HOLOGRAM_SIZE[0]  # Tamaño de pixel en μm


def generate_random_array(grid_size, occupancy):
    total_sites = grid_size * grid_size
    atom_count = int(total_sites * occupancy)
    indices = np.random.choice(total_sites, atom_count, replace=False)
    array = np.zeros(total_sites, dtype=bool)
    array[indices] = True
    return array.reshape((grid_size, grid_size))

def get_atom_positions(array):
    return np.argwhere(array == 1)

def grid_to_physical(positions):
    return positions * GRID_SPACING

def get_target_positions(grid_size, target_size):
    offset = (grid_size - target_size) // 2
    i_coords, j_coords = np.mgrid[0:target_size, 0:target_size]
    positions = np.column_stack([(i_coords + offset).ravel(), (j_coords + offset).ravel()])
    return positions

def compute_cost_matrix_scipy(atoms, targets, alpha=1.0):
    if alpha == 1.0:
        return cdist(targets, atoms, metric='euclidean')
    elif alpha == 2.0:
        return cdist(targets, atoms, metric='sqeuclidean')
    else:
        distances = cdist(targets, atoms, metric='euclidean')
        return distances ** alpha

def find_discarded_atoms_vectorized(all_atoms, selected_atoms):
    if len(selected_atoms) == 0:
        return all_atoms
    all_expanded = all_atoms[:, np.newaxis, :]
    selected_expanded = selected_atoms[np.newaxis, :, :]
    equal_elements = (all_expanded == selected_expanded)
    equal_rows = np.all(equal_elements, axis=2)
    is_selected = np.any(equal_rows, axis=1)
    return all_atoms[~is_selected]

def select_closest_atoms_vectorized(atoms, center, n_targets):
    distances = np.linalg.norm(atoms - center, axis=1)
    sorted_indices = np.argpartition(distances, min(n_targets-1, len(distances)-1))[:n_targets]
    return atoms[sorted_indices], sorted_indices

def assign_subregion(atoms, targets, alpha):
    # Convertir a coordenadas físicas para el cálculo de costos
    atoms_physical = grid_to_physical(atoms)
    targets_physical = grid_to_physical(targets)
    
    cost_matrix = compute_cost_matrix_scipy(atoms_physical, targets_physical, alpha)
    _, col_ind, _ = lapjv(cost_matrix)
    assigned_atoms = atoms[col_ind]
    return assigned_atoms

def create_target_from_positions(positions, hologram_size, grid_size):
    """
    Crear el target con píxeles binarios.
    Valor 1 donde hay un átomo, valor 0 donde no hay.
    """
    target = np.zeros(hologram_size)
    
    # Convertir posiciones de grid a coordenadas físicas (micrómetros)
    positions_physical = grid_to_physical(positions)
    
    # Calcular el offset para centrar el patrón de átomos en el holograma
    system_center_um = SYSTEM_SIZE_UM / 2
    hologram_center_um = HOLOGRAM_PHYSICAL_SIZE_UM / 2
    
    # Offset en micrómetros para centrar el patrón
    offset_x_um = hologram_center_um - system_center_um
    offset_y_um = hologram_center_um - system_center_um
    
    for pos_phys in positions_physical:
        # Convertir coordenadas físicas a coordenadas de píxeles del holograma
        
        x_um = pos_phys[1] + offset_x_um  # Coordenada X en micrómetros
        y_um = pos_phys[0] + offset_y_um  # Coordenada Y en micrómetros
        
        # Convertir a píxeles (Redondear a la posición de píxel más cercana)

        px = int(round(x_um / HOLOGRAM_PIXEL_SIZE_UM))
        py = int(round(y_um / HOLOGRAM_PIXEL_SIZE_UM))
        
        target[py, px] = 1.0  # Píxel binario: 1 donde hay átomo
    
    return target


def print_scaling_info():
    print(f"Sistema de átomos:")
    print(f"  - Tamaño del grid: {GRID_SIZE} x {GRID_SIZE}")
    print(f"  - Espaciado entre átomos: {GRID_SPACING} μm")
    print(f"  - Tamaño total del sistema: {SYSTEM_SIZE_UM} μm")
    print(f"  - Diámetro del átomo: {ATOM_DIAMETER} μm")
    
    print(f"\nHolograma:")
    print(f"  - Tamaño en píxeles: {HOLOGRAM_SIZE[0]} x {HOLOGRAM_SIZE[1]}")
    print(f"  - Tamaño físico: {HOLOGRAM_PHYSICAL_SIZE_UM} μm")
    print(f"  - Tamaño de píxel: {HOLOGRAM_PIXEL_SIZE_UM:.3f} μm/píxel")
    
    # Información sobre representación binaria
    atoms_per_pixel = (HOLOGRAM_PIXEL_SIZE_UM / GRID_SPACING)**2
    print(f"  - Resolución espacial: {atoms_per_pixel:.3f} posiciones atómicas por píxel")
    
    

def compute_hologram(positions, hologram_size, grid_size, iterations=10, initial_phase=None):
    """Calcular el holograma para un conjunto de posiciones atómicas."""
    
    # Crear el target basado en las posiciones
    target = create_target_from_positions(positions, hologram_size, grid_size)
    
    # Inicializar el holograma especificando explícitamente slm_shape
    if initial_phase is not None:
        # Usar la fase inicial proporcionada
        holo = Hologram(target=target, phase=initial_phase, slm_shape=hologram_size)

    else:
        # Usar inicialización por defecto (aleatoria)
        holo = Hologram(target=target, slm_shape=hologram_size)
          
    # Optimizar el holograma
    holo.optimize(method="WGS-Kim", maxiter=iterations, verbose=False) #stat_groups=["computational"]
    
    # Obtener resultados
    phase = holo.get_phase()
    recovered_intensity = math.sqrt(HOLOGRAM_SIZE[0] * HOLOGRAM_SIZE[1]) * np.abs(holo.get_farfield())**2
    #holo.plot_stats()
    
    return phase, recovered_intensity, target


def precompute_holograms(selected_atoms, final_targets, grid_size, steps):
    """Pre-computar todos los hologramas para cada paso de la interpolación."""
    holograms = []
    intensities = []
    targets = []
    
    print("Pre-computando hologramas...")
    
    start_time = time.perf_counter()
    
    previous_phase = None  # Para almacenar la fase del paso anterior
    
    for step in range(steps + 1):
        t = step / steps
        interp_pos = (1 - t) * selected_atoms + t * final_targets
        
        print(f"Calculando paso {step}/{steps}...")
        
        # Calcular holograma para esta configuración
        if step == 0:
            # Primer paso: usar inicialización por defecto
            phase, intensity, target = compute_hologram(
                interp_pos, HOLOGRAM_SIZE, grid_size, 
                HOLOGRAM_ITERATIONS, initial_phase=None
            )
            
        else:
            # Pasos posteriores: usar la fase del paso anterior como inicial
            phase, intensity, target = compute_hologram(
                interp_pos, HOLOGRAM_SIZE, grid_size, 
                HOLOGRAM_ITERATIONS, initial_phase=previous_phase
            )
            
        
        holograms.append(phase)
        intensities.append(intensity)
        targets.append(target)
        
        # Guardar la fase actual para usar en el siguiente paso
        previous_phase = phase.copy()  # Importante: hacer una copia
       
    
    end_time = time.perf_counter()
    hologram_time = end_time - start_time
    print(f"Tiempo total de cálculo de hologramas: {hologram_time:.2f} segundos")
    
    return holograms, intensities, targets, hologram_time


def plot_complete_trajectory_slider(selected_atoms, final_targets, grid_size, steps, 
                                   discarded_atoms, assign_time, holograms, intensities, 
                                   targets, hologram_time):
    """Mostrar el slider con las posiciones atómicas y los hologramas en una sola fila."""

    # Convertir a coordenadas físicas
    selected_atoms_phys = grid_to_physical(selected_atoms)
    final_targets_phys = grid_to_physical(final_targets)
    discarded_atoms_phys = grid_to_physical(discarded_atoms)
    
    def get_plot_coords(physical_pos):
        if physical_pos.ndim == 1:
            return physical_pos[1], SYSTEM_SIZE_UM - GRID_SPACING - physical_pos[0]
        return physical_pos[:, 1], SYSTEM_SIZE_UM - GRID_SPACING - physical_pos[:, 0]

    # ---  Ventana 3 paneles ------------------------------------
    plt.rcParams['figure.dpi'] = 120
    fig = plt.figure(figsize=(24, 6))

    # ---  Una fila, 3 columnas --------------------------------------
    gs = GridSpec(1, 3, figure=fig,        # 1 fila, 3 columnas
                  left=0.02, right=0.98,    # casi sin márgenes laterales
                  top=0.85, bottom=0.15,    # espacio arriba para el título y abajo para el slider
                  wspace=0.05)              # hueco fino entre paneles

    ax2 = fig.add_subplot(gs[0, 0]); ax2.set_title("Target Intensity")
    ax3 = fig.add_subplot(gs[0, 1]); ax3.set_title("Recovered Intensity")
    ax4 = fig.add_subplot(gs[0, 2]); ax4.set_title("Hologram Phase")
    for ax in [ax2, ax3, ax4]:
        ax.tick_params(axis='both', labelsize=8)  

    num_divisions=5
    xticks=np.linspace(0, HOLOGRAM_SIZE[1], num_divisions, dtype=int)

    # --- Configurar los paneles de hologramas con ejes y colorbars ---------------------------
    
    # Panel 1: Target Intensity
    im2 = ax2.imshow(targets[0], cmap='inferno', extent=[0, HOLOGRAM_SIZE[1], HOLOGRAM_SIZE[0], 0])
    ax2.set_title("Target Intensity", fontsize=14, fontweight='bold')
    ax2.set_xticks(xticks)
    ax2.set_yticks([])
   
    # Colorbar para target intensity
    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8, aspect=20)
    cbar2.ax.tick_params(labelsize=8)

    # Panel 2: Recovered Intensity  
    im3 = ax3.imshow(intensities[0], cmap='inferno', extent=[0, HOLOGRAM_SIZE[1], HOLOGRAM_SIZE[0], 0])
    ax3.set_title("Recovered Intensity", fontsize=14, fontweight='bold')
    ax3.set_xticks(xticks)
    ax3.set_yticks([])

    # Colorbar para recovered intensity
    cbar3 = plt.colorbar(im3, ax=ax3, shrink=0.8, aspect=20)
    cbar3.ax.tick_params(labelsize=8)

    # Panel 3: Hologram Phase
    im4 = ax4.imshow(holograms[0], cmap='twilight', extent=[0, HOLOGRAM_SIZE[1], HOLOGRAM_SIZE[0], 0])
    ax4.set_title("Hologram Phase", fontsize=14, fontweight='bold')
    ax4.set_xticks(xticks)
    ax4.set_yticks([])

    # Colorbar para phase con etiquetas en radianes
    cbar4 = plt.colorbar(im4, ax=ax4, shrink=0.8, aspect=20)
    cbar4.ax.tick_params(labelsize=8)

    # Configurar ticks de la colorbar para mostrar valores en π
    phase_ticks = np.linspace(0, 2*np.pi, 11)
    phase_labels = [f"{t/np.pi:.1f}π" for t in phase_ticks]
    cbar4.set_ticks(phase_ticks)
    cbar4.set_ticklabels(phase_labels)

    # --- Texto de tiempos  --------------------
    fig.suptitle(f"Asignación: {assign_time:.4f}s | Hologramas: {hologram_time:.2f}s",
                 fontsize=13, fontweight='bold')

    # --- Slider a lo largo de toda la fila ------------------------------
    ax_slider = plt.axes([0.07, 0.04, 0.86, 0.05])  # (x, y, ancho, alto) en coords. fig.
    slider = Slider(ax_slider, 'Paso', 0, steps, valinit=0, valstep=1)

    def update(val):
        step = int(slider.val)
       
        # Actualizar hologramas
        im2.set_data(targets[step])
        im3.set_data(intensities[step])
        im4.set_data(holograms[step])
        fig.canvas.draw_idle()
    
    slider.on_changed(update)
    update(0)
    
    plt.show()

def main():
    print("Generando array inicial...")
    array = generate_random_array(GRID_SIZE, OCCUPANCY)
    atoms = get_atom_positions(array)
    targets = get_target_positions(GRID_SIZE, TARGET_SIZE)
    
    print(f"Átomos encontrados: {len(atoms)}")
    print(f"Posiciones objetivo: {len(targets)}")
    
    if len(atoms) < len(targets):
        print("No hay suficientes átomos para llenar el array sin defectos.")
        return
    
    mid = GRID_SIZE // BLOCKS
    quadrants = []
    for i in range(BLOCKS):
        for j in range(BLOCKS):
            target_mask = (
                (targets[:, 0] >= i * mid) & (targets[:, 0] < (i + 1) * mid) &
                (targets[:, 1] >= j * mid) & (targets[:, 1] < (j + 1) * mid)
            )
            targets_q = targets[target_mask]
            if len(targets_q) > 0:
                center_q = targets_q.mean(axis=0)
            
            atom_mask = (
                (atoms[:, 0] >= i * mid) & (atoms[:, 0] < (i + 1) * mid) &
                (atoms[:, 1] >= j * mid) & (atoms[:, 1] < (j + 1) * mid)
            )
            atoms_q = atoms[atom_mask]
            
            if len(atoms_q) >= len(targets_q):
                selected_q, _ = select_closest_atoms_vectorized(atoms_q, center_q, len(targets_q))
                quadrants.append((selected_q, targets_q))
            else:
                print(f"Cuadrante ({i},{j}) no tiene suficientes átomos. " + 
                      f"Tiene {len(atoms_q)} y necesita {len(targets_q)}.")
                return
    
    print("Ejecutando asignaciones en paralelo...")
    start_time = time.perf_counter()
    with ThreadPoolExecutor(max_workers=BLOCKS**2) as executor:
        results = list(executor.map(lambda args: assign_subregion(*args, ALPHA), quadrants))
    end_time = time.perf_counter()
    assign_time = end_time - start_time
    print(f"Tiempo total de asignación paralela: {assign_time:.6f} segundos")
    
    assigned_atoms_blocks = []
    targets_blocks = []
    for (atoms_q, targets_q), assigned_q in zip(quadrants, results):
        assigned_atoms_blocks.append(assigned_q)
        targets_blocks.append(targets_q)
    
    assigned_atoms = np.vstack(assigned_atoms_blocks)
    targets_ordered = np.vstack(targets_blocks)
  
    discarded_atoms = find_discarded_atoms_vectorized(atoms, assigned_atoms)
    
    print_scaling_info()  
    
    # Pre-computar hologramas
    holograms, intensities, target_images, hologram_time = precompute_holograms(
        assigned_atoms, targets_ordered, GRID_SIZE, STEPS
    )
    
    # Mostrar el slider completo con 4 paneles
    plot_complete_trajectory_slider(
        assigned_atoms, targets_ordered, GRID_SIZE, STEPS,
        discarded_atoms, assign_time, holograms, intensities, 
        target_images, hologram_time
    )

if __name__ == "__main__":
    main()