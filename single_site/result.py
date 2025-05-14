import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

class SingleSiteResult:
    def __init__(self, data):
        self.data = data

    def compute_energy(self):
        return self.data.get("eigenvalue", None)

    def summary(self):
        print("=== Simulation Summary ===")
        print(f"Ground state energy: {self.compute_energy()}")
        for k, v in self.data.get("meta", {}).items():
            print(f"{k}: {v}")

    def plot_energy_surface(self, cmap=None):
        Qx = self.data.get("Qx")
        Qy = self.data.get("Qy")
        energy = self.data.get("energy_map")

        if Qx is None or Qy is None or energy is None:
            print("This result does not contain a 2D energy map.")
            return

        if cmap is None:
            colors = [
                (0.0, "black"), (0.15, "yellow"), (0.25, "orange"),
                (0.35, "red"), (0.5, "magenta"), (0.65, "blue"),
                (0.85, "cyan"), (1.0, "white")
            ]
            cmap = LinearSegmentedColormap.from_list("custom_map", colors)

        fig, ax = plt.subplots()
        c = ax.pcolormesh(Qx, Qy, energy, cmap=cmap, shading='gouraud')
        fig.colorbar(c, ax=ax)
        ax.set_xlabel("Qx")
        ax.set_ylabel("Qy")
        ax.set_title("Energy Landscape")
        plt.show()
