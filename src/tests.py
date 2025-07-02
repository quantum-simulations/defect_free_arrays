import os
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from data_generation import DataGenerator
from cost_matrix import CostMatrix, Solver
from atom_selection import AtomSelector


class BenchmarkTester:
    def __init__(self, grid_size=120, alpha=2, save_dir="assets/benchmark_results"):
        self.grid_size = grid_size
        self.alpha = alpha
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir


    def benchmark_time_heatmap_sizes(self, min_size=2, max_size=20, runs=20):
        sizes = list(range(min_size, max_size + 1))
        heatmap = np.zeros((len(sizes), len(sizes)))

        filename = os.path.join(self.save_dir, f"heatmap_min{min_size}_max{max_size}_runs{runs}reservoir{self.grid_size}.csv")
        with open(filename, "w") as f:
            f.write("res atoms,target atoms,assignment time (ms)\n")

        for i, reservoir_size in enumerate(tqdm(sizes, desc="Reservoir size")):
            for j, target_size in enumerate(sizes):
                if target_size > reservoir_size:
                    heatmap[i, j] = np.nan
                    continue

                times = []
                for _ in range(runs):
                    data_gen = DataGenerator(grid_size=self.grid_size, target_size=target_size)
                    atom_positions1 = data_gen.get_atom_positions()
                    atom_positions2 = data_gen.get_target_positions()
                    center = data_gen.target_center
                    ntargets = data_gen.target_sites

                    atom_selector = AtomSelector(atom_positions1, atom_positions2)
                    selected_atoms, _ = atom_selector.select_closest_atoms(center, ntargets)
                    atom_selector.find_discarded_atoms(selected_atoms)

                    cost_matrix = CostMatrix(selected_atoms, atom_positions2, alpha=self.alpha).compute_cost_matrix()
                    solver = Solver(cost_matrix)

                    start = time.perf_counter()
                    solver.lapjv()
                    end = time.perf_counter()

                    times.append((end - start) * 1000)

                heatmap[i, j] = np.mean(times)

                # Write the result for this (i, j) immediately
                value = heatmap[i, j]
                if not np.isnan(value):
                    x = reservoir_size ** 2
                    y = target_size ** 2
                    with open(filename, "a") as f:
                        f.write(f"{x},{y},{value:.6f}\n")
        return sizes, heatmap

    def benchmark_time_heatmap_occupancy_targetsize(self, reservoir_size, target_range, occupancy_range, runs=20):
        heatmap = np.zeros((len(target_range), len(occupancy_range)))

        filename = os.path.join(self.save_dir, f"occupancy_map_res{reservoir_size}_runs{runs}.csv")
        with open(filename, "w") as f:
            f.write("target atoms,occupancy,assignment time (ms)\n")

        for i, target_size in enumerate(tqdm(target_range, desc="Target size")):
            for j, occupancy in enumerate(occupancy_range):
                times = []
                for _ in range(runs):
                    data_gen = DataGenerator(
                        grid_size=self.grid_size,
                        target_size=target_size,
                        occupancy=occupancy
                    )
                    atom_positions1 = data_gen.get_atom_positions()
                    atom_positions2 = data_gen.get_target_positions()
                    center = data_gen.target_center
                    ntargets = data_gen.target_sites

                    atom_selector = AtomSelector(atom_positions1, atom_positions2)
                    selected_atoms, _ = atom_selector.select_closest_atoms(center, ntargets)
                    atom_selector.find_discarded_atoms(selected_atoms)

                    cost_matrix = CostMatrix(selected_atoms, atom_positions2, alpha=self.alpha).compute_cost_matrix()
                    solver = Solver(cost_matrix)

                    start = time.perf_counter()
                    solver.lapjv()
                    end = time.perf_counter()
                    times.append((end - start) * 1000)

                heatmap[i, j] = np.mean(times)

                # Write the result for this (i, j) immediately
                value = heatmap[i, j]
                if not np.isnan(value):
                    x = target_size ** 2
                    y = occupancy
                    with open(filename, "a") as f:
                        f.write(f"{x},{y},{value:.6f}\n")
        return target_range, occupancy_range, heatmap

    def benchmark_assignment_time_vs_target_atoms(self, reservoir_size, target_range, occupancy=1.0, runs=20):
        means = []
        stds = []

        filename = os.path.join(self.save_dir, f"target_vs_time_res{reservoir_size}_runs{runs}.csv")
        with open(filename, "w") as f:
            f.write("x,y,error\n")

        for target_size in tqdm(target_range, desc="Target size"):
            times = []
            for _ in range(runs):
                data_gen = DataGenerator(
                    grid_size=self.grid_size,
                    target_size=target_size,
                    occupancy=occupancy
                )
                atom_positions1 = data_gen.get_atom_positions()
                atom_positions2 = data_gen.get_target_positions()
                center = data_gen.target_center
                ntargets = data_gen.target_sites

                atom_selector = AtomSelector(atom_positions1, atom_positions2)
                selected_atoms, _ = atom_selector.select_closest_atoms(center, ntargets)
                atom_selector.find_discarded_atoms(selected_atoms)

                cost_matrix = CostMatrix(selected_atoms, atom_positions2, alpha=self.alpha).compute_cost_matrix()
                solver = Solver(cost_matrix)

                start = time.perf_counter()
                solver.lapjv()
                end = time.perf_counter()
                times.append((end - start) * 1000)

            mean = np.mean(times)
            std = np.std(times)
            means.append(mean)
            stds.append(std)

            # Write the result for this target_size immediately
            x = target_size ** 2
            y = mean
            with open(filename, "a") as f:
                f.write(f"{x},{y},{std:.6f}\n")

        return target_range, means, stds

class BenchmarkPlotter:
    def __init__(self):
        pass

    def time_heatmap_size(self, sizes, heatmap, title="Assignment Time Heatmap",
                     xlabel="Target size²", ylabel="Reservoir size²", save_path=None):
        plt.figure(figsize=(10, 8))
        masked = np.ma.masked_invalid(heatmap)
        extent = [sizes[0] ** 2, sizes[-1] ** 2, sizes[0] ** 2, sizes[-1] ** 2]

        c = plt.imshow(masked, interpolation='nearest', cmap='viridis',
                       origin='lower', extent=extent, aspect='auto')

        plt.colorbar(c, label="Mean assignment time (ms)")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(False)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        # plt.show()

    def time_heatmap_occupancy(self, target_range, occupancy_range, heatmap,
                           title="Assignment Time vs Target and Occupancy",
                           save_path=None):
        plt.figure(figsize=(10, 8))
        masked = np.ma.masked_invalid(heatmap)
        extent = [occupancy_range[0], occupancy_range[-1], target_range[0] ** 2, target_range[-1] ** 2]

        c = plt.imshow(masked, interpolation='nearest', cmap='plasma',
                       origin='lower', extent=extent, aspect='auto')

        plt.colorbar(c, label="Mean assignment time (ms)")
        plt.xlabel("Occupancy")
        plt.ylabel("Number of atoms in target array")
        plt.title(title)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        # plt.show()

    def time_vs_size(self, target_range, means, stds,
                             title="Assignment Time vs Target Size",
                             xlabel="Number of atoms in target array",
                             ylabel="Mean assignment time (ms)",
                             save_path=None):
        x = [t ** 2 for t in target_range]

        plt.figure(figsize=(10, 6))
        plt.errorbar(x, means, yerr=stds, fmt='-o', capsize=5, label='Mean ± Std')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        # plt.show()


