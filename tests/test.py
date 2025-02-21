"""
Solar System Simulation using SPICE (Barycentric) - Optimized Version
-----------------------------------------------
This script simulates planetary orbits around the Solar System Barycenter using NASA's SPICE toolkit.
Author: [Your Name]
Date: February 2025

Requirements:
- SpiceyPy
- NumPy
- Matplotlib
- SciPy
"""

import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import spiceypy as spice
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import datetime
from typing import List, Tuple, Dict
from pathlib import Path
import logging

# Configuration remains the same
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Physical constants remain the same
G = 6.67430e-11
AU = 149597870700
SECONDS_PER_DAY = 86400
STANDARD_EPOCH = "2000-01-01T12:00:00"

# Body properties remain the same
BODY_PROPERTIES = {
    'SUN': {'color': '#FFD700', 'size': 200, 'zorder': 10},
    'MERCURY': {'color': '#A0522D', 'size': 50, 'zorder': 11},
    'VENUS': {'color': '#DEB887', 'size': 95, 'zorder': 12},
    'EARTH': {'color': '#4169E1', 'size': 100, 'zorder': 13},
    'MARS': {'color': '#CD5C5C', 'size': 75, 'zorder': 14},
    'JUPITER': {'color': '#DAA520', 'size': 150, 'zorder': 15},
    'SATURN': {'color': '#B8860B', 'size': 140, 'zorder': 16},
    'URANUS': {'color': '#87CEEB', 'size': 120, 'zorder': 17},
    'NEPTUNE': {'color': '#1E90FF', 'size': 120, 'zorder': 18}
}

class CelestialBody:
    """Represents a celestial body using SPICE data."""

    def __init__(self, naif_id: int, name: str, system: 'SolarSystem'):
        """Initialize a celestial body with SPICE data."""
        self.naif_id = naif_id
        self.name = name.upper()
        self.system = system

        try:
            # Cache physical parameters to avoid repeated SPICE calls
            if naif_id != 10:
                radii_data = spice.bodvcd(naif_id, "RADII", 3)
                self.radii = np.mean(radii_data[1]) * 1000
            else:
                self.radii = 696340000

            # Cache GM value
            if naif_id == 10:
                self.GM = G * 1.98847e30
            else:
                self.GM = spice.bodvcd(naif_id, "GM", 1)[1][0] * 1e9

            # Get visualization properties
            props = BODY_PROPERTIES.get(self.name,
                                      {'color': '#FFFFFF', 'size': 50, 'zorder': 10})
            self.color = props['color']
            self.vis_size = props['size']
            self.zorder = props['zorder']

        except spice.support_types.SpiceyError as e:
            logger.error(f"SPICE error for {self.name}: {str(e)}")
            raise

class SolarSystem:
    """Manages the solar system simulation environment with interactive 3D visualization."""

    def __init__(self,
                 start_date: str = STANDARD_EPOCH,
                 duration_days: int = 365,
                 step_days: int = 1,
                 animation_skip: int = 1):
        """Initialize solar system simulation."""
        self.bodies: List[CelestialBody] = []

        # Convert start date to ET once and store
        self.start_et = spice.str2et(start_date)
        self.start_date = datetime.datetime.strptime(start_date, "%Y-%m-%dT%H:%M:%S")

        # Pre-calculate time array in ET format
        self.duration = duration_days * SECONDS_PER_DAY
        self.step = step_days * SECONDS_PER_DAY
        self.t = np.arange(0, self.duration, self.step)
        self.et_array = self.start_et + self.t
        self.num_frames = len(self.t)

        # Animation properties
        self.trail_length = 50
        self.animation_skip = animation_skip

        # Setup interactive 3D visualization
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(10, 8))
        self.ax_anim = self.fig.add_subplot(111, projection='3d')
        self.ax_anim.set_title("Interactive 3D Solar System Motion")
        self.ax_anim.set_xlabel("X [AU]")
        self.ax_anim.set_ylabel("Y [AU]")
        self.ax_anim.set_zlabel("Z [AU]")

        # Dictionary to store visualization elements for each body.
        # Each entry will be a tuple: (line, point, text)
        self.vis_elements = {}

    def _setup_axes(self):
        """Configure the appearance of the animation axis."""
        self.ax_anim.set_title("Solar System Motion")
        self.ax_anim.set_xlabel("X [AU]")
        self.ax_anim.set_ylabel("Y [AU]")
        self.ax_anim.grid(True, alpha=0.3)
        self.ax_anim.set_aspect('equal')
        self.timer_text = self.ax_anim.text(0.02, 0.95, '',
                                          transform=self.ax_anim.transAxes,
                                          color='white')

    def add_body(self, naif_id: int, name: str) -> None:
        """Add a celestial body to the simulation."""
        body = CelestialBody(naif_id, name, self)
        self.bodies.append(body)
        logger.info(f"Added {name} to simulation")

    def simulate(self) -> Dict:
        """Get position data for all bodies relative to SSB using vectorized SPICE calls,
        with performance optimization for large time spans via chunking.
        """
        results = {}
        CHUNK_SIZE = 5000  # Adjust based on memory/performance
        num_chunks = (self.num_frames + CHUNK_SIZE - 1) // CHUNK_SIZE

        for body in self.bodies:
            states_list = []
            for chunk in range(num_chunks):
                start_idx = chunk * CHUNK_SIZE
                end_idx = min(start_idx + CHUNK_SIZE, self.num_frames)
                chunk_et = self.et_array[start_idx:end_idx]

                positions, lt = spice.spkpos(
                    targ=str(body.naif_id),
                    et=chunk_et,
                    ref="ECLIPJ2000",
                    abcorr="NONE",
                    obs="0"  # SSB
                )

                # Convert positions from km to m then to AU and transpose to shape (3, num_frames)
                states_chunk = (np.array(positions).T * 1000) / AU
                states_list.append(states_chunk)

            # Concatenate all chunks along the time axis.
            states = np.concatenate(states_list, axis=1)
            results[body.name] = states

            # Plot the full orbit path (static) in 3D.
            self.ax_anim.plot(states[0], states[1], states[2],
                              color=body.color,
                              alpha=0.3)

            # Initialize interactive elements:
            #  - A line for the trail (empty initially)
            #  - A point for the current position
            #  - A text label for the body name
            line, = self.ax_anim.plot([], [], [], '-', color=body.color,
                                        alpha=0.8, zorder=body.zorder)
            point, = self.ax_anim.plot([], [], [], 'o', color=body.color,
                                         markersize=body.vis_size / 10, zorder=body.zorder)
            text = self.ax_anim.text(0, 0, 0, body.name, color='white')
            self.vis_elements[body.name] = (line, point, text)

        # Add the Solar System Barycenter (SSB) marker.
        self.ax_anim.plot([0], [0], [0], marker='+', color='white',
                          markersize=10, label='SSB', zorder=1000)
        self.ax_anim.legend(loc='upper right')

        # Set 3D axis limits based on the maximum position from all bodies.
        max_pos = max(np.max(np.abs(states)) for states in results.values())
        limit = max_pos * 1.2
        self.ax_anim.set_xlim3d(-limit, limit)
        self.ax_anim.set_ylim3d(-limit, limit)
        self.ax_anim.set_zlim3d(-limit, limit)

        self.results = results  # Store simulation results for use in animation
        return results

    def animate(self, interval: int = 50) -> FuncAnimation:
        """Create an interactive 3D animation of planetary motion."""
        def update(frame):
            actual_frame = frame * self.animation_skip
            elements = []
            for body in self.bodies:
                states = self.results[body.name]  # shape: (3, num_frames)
                # Get trail data from start to current frame.
                x_data = states[0, :actual_frame + 1]
                y_data = states[1, :actual_frame + 1]
                z_data = states[2, :actual_frame + 1]
                line, point, text = self.vis_elements[body.name]

                # Update trail (line) data.
                line.set_data(x_data, y_data)
                line.set_3d_properties(z_data)

                # Update current position (point).
                current_x = states[0, actual_frame]
                current_y = states[1, actual_frame]
                current_z = states[2, actual_frame]
                point.set_data([current_x], [current_y])
                point.set_3d_properties([current_z])

                # Update text label position.
                text.set_position((current_x, current_y))
                text.set_3d_properties(current_z, zdir="z")

                elements.extend([line, point, text])
            return elements

        total_frames = self.num_frames // self.animation_skip
        ani = FuncAnimation(self.fig, update,
                            frames=range(total_frames),
                            interval=interval,
                            blit=True)

        self.fig.suptitle("Interactive 3D Solar System Bodies Relative to SSB", y=0.95)
        plt.tight_layout()
        return ani

def main():
    """Main execution function."""
    try:
        kernel_path = Path("../data/ephemeris/spice/meta/metakernel.tm")
        if not kernel_path.exists():
            raise FileNotFoundError(f"SPICE kernel not found at {kernel_path}")

        spice.furnsh(str(kernel_path))
        logger.info("SPICE kernels loaded successfully")

        solar_system = SolarSystem(
            start_date="2000-01-01T00:00:00",
            duration_days=730,
            step_days=1
        )

        bodies = [
            (10, "Sun"),
            (199, "Mercury"),
            (299, "Venus"),
            (399, "Earth"),
            (499, "Mars"),
        ]

        for naif_id, name in bodies:
            solar_system.add_body(naif_id, name)

        results = solar_system.simulate()
        animation = solar_system.animate(interval=50)
        plt.show()

    except Exception as e:
        logger.error(f"Simulation failed: {str(e)}")
        raise
    finally:
        spice.kclear()

if __name__ == "__main__":
    main()
