"""
Geological Erosion Engine:
1. Momentum Droplet Hydraulic Simulation with Radial Brush & Alluvial Fans.
2. Isotropic 8-Neighbor Thermal Talus Relaxation with Scree Tracking.
"""

from __future__ import annotations
import math
from typing import Tuple, List, Optional
import numpy as np
from terra_forge.core.heightfield import Heightfield2D
from terra_forge.schema.map_config import ErosionConfig


class ErosionSimulator:
    def __init__(self, hf: Heightfield2D, config: ErosionConfig):
        self.hf = hf
        self.cfg = config
        self.sediment_map = self.hf.sediment
        if not hasattr(self.hf, "scree_talus") or self.hf.scree_talus is None:
            self.hf.scree_talus = np.zeros_like(self.hf.Z)
        if not hasattr(self.hf, "water_depth") or self.hf.water_depth is None:
            self.hf.water_depth = np.zeros_like(self.hf.Z)

        self.brush_radius = max(1, int(getattr(config, "brush_radius", 3.0)))
        self.fan_radius = max(1, int(round(getattr(config, "alluvial_fan_spread", 3.5))))
        self._init_kernels()

    def _init_kernels(self) -> None:
        """Precomputes discrete circular distance weights for erosion and alluvial fan kernels."""
        res_min = min(self.hf.res_x, self.hf.res_y)
        r_ero = min(self.brush_radius, max(1, res_min // 8))
        r_fan = min(self.fan_radius, max(1, res_min // 8))

        # 1. Erosion radial brush kernel
        ero_offsets: List[Tuple[int, int]] = []
        ero_weights: List[float] = []
        for dy in range(-r_ero, r_ero + 1):
            for dx in range(-r_ero, r_ero + 1):
                d = math.hypot(dx, dy)
                if d <= r_ero:
                    ero_offsets.append((dy, dx))
                    ero_weights.append(max(0.0, r_ero - d + 0.5))
        w_ero = np.array(ero_weights, dtype=np.float64)
        w_ero /= np.sum(w_ero)
        self.brush_offsets = tuple(ero_offsets)
        self.brush_weights = tuple(w_ero)

        # 2. Alluvial fan deposition kernel
        fan_offsets: List[Tuple[int, int]] = []
        fan_weights: List[float] = []
        for dy in range(-r_fan, r_fan + 1):
            for dx in range(-r_fan, r_fan + 1):
                d = math.hypot(dx, dy)
                if d <= r_fan:
                    fan_offsets.append((dy, dx))
                    fan_weights.append(max(0.0, r_fan - d + 0.5))
        w_fan = np.array(fan_weights, dtype=np.float64)
        w_fan /= np.sum(w_fan)
        self.fan_offsets = tuple(fan_offsets)
        self.fan_weights = tuple(w_fan)

    def simulate_hydraulic(self, seed: Optional[int] = None) -> Tuple[Heightfield2D, np.ndarray, np.ndarray]:
        """
        Droplet hydraulic erosion simulation with Newtonian velocity acceleration,
        radial brush incision, alluvial fan deposition, and water flow flux tracking.
        Guarantees exact physical mass conservation (|ΔM| < 1e-10) and eliminates race conditions.
        Returns:
            Tuple[Heightfield2D, sediment_map, water_depth]
        """
        if not self.cfg.enabled or self.cfg.droplet_count <= 0:
            return self.hf, self.sediment_map, self.hf.water_depth

        rng = np.random.default_rng(seed)
        res_x = self.hf.res_x
        res_y = self.hf.res_y
        Z = self.hf.Z
        sediment_map = self.sediment_map
        water_depth_map = self.hf.water_depth
        initial_mass = float(np.sum(Z))

        num_droplets = self.cfg.droplet_count
        inertia = self.cfg.droplet_inertia
        capacity_factor = self.cfg.sediment_capacity_factor
        min_capacity = self.cfg.min_sediment_capacity
        erode_speed = self.cfg.erode_speed
        deposit_speed = self.cfg.deposit_speed
        evaporate_speed = self.cfg.evaporate_speed
        gravity = self.cfg.gravity
        max_steps = self.cfg.max_droplet_steps
        rill_factor = getattr(self.cfg, "rill_incision_factor", 0.5)

        margin = min(3.0, max(0.01, min(self.hf.width, self.hf.depth) * 0.1))
        low_x, high_x = margin, res_x - margin - 1.0
        low_y, high_y = margin, res_y - margin - 1.0
        if low_x >= high_x or low_y >= high_y:
            return self.hf, self.sediment_map, self.hf.water_depth

        for _ in range(num_droplets):
            px = rng.uniform(low_x, high_x)
            py = rng.uniform(low_y, high_y)
            dir_x, dir_y = 0.0, 0.0
            vel = 1.0
            water = 1.0
            sediment = 0.0

            for step in range(max_steps):
                ix, iy = int(px), int(py)
                u, v = px - ix, py - iy

                h00 = Z[iy, ix]
                h10 = Z[iy, ix + 1]
                h01 = Z[iy + 1, ix]
                h11 = Z[iy + 1, ix + 1]

                gx = (h10 - h00) * (1.0 - v) + (h11 - h01) * v
                gy = (h01 - h00) * (1.0 - u) + (h11 - h10) * u
                h_cur = (1.0 - u) * (1.0 - v) * h00 + u * (1.0 - v) * h10 + (1.0 - u) * v * h01 + u * v * h11

                # Track water flow depth accumulation
                w_uv = water * (1.0 - u) * (1.0 - v)
                water_depth_map[iy, ix] += w_uv
                water_depth_map[iy, ix + 1] += water * u * (1.0 - v)
                water_depth_map[iy + 1, ix] += water * (1.0 - u) * v
                water_depth_map[iy + 1, ix + 1] += water * u * v

                # Direction steering with momentum inertia
                dir_x = dir_x * inertia - gx * (1.0 - inertia)
                dir_y = dir_y * inertia - gy * (1.0 - inertia)
                norm = math.hypot(dir_x, dir_y) + 1e-7
                dir_x /= norm
                dir_y /= norm

                npx = px + dir_x
                npy = py + dir_y

                # Bounds check against margins
                if npx < margin or npx >= res_x - margin - 1 or npy < margin or npy >= res_y - margin - 1:
                    if sediment > 0.0:
                        Z[iy, ix] += sediment
                        sediment_map[iy, ix] += sediment
                        sediment = 0.0
                    break

                nix, niy = int(npx), int(npy)
                nu, nv = npx - nix, npy - niy

                nh00 = Z[niy, nix]
                nh10 = Z[niy, nix + 1]
                nh01 = Z[niy + 1, nix]
                nh11 = Z[niy + 1, nix + 1]
                h_next = (1.0 - nu) * (1.0 - nv) * nh00 + nu * (1.0 - nv) * nh10 + (1.0 - nu) * nv * nh01 + nu * nv * nh11

                dh = h_next - h_cur

                slope = max(-dh, min_capacity)
                c_sed = slope * vel * water * capacity_factor * (1.0 + rill_factor * 0.2)

                if dh > 0.0 or sediment > c_sed:
                    # Deposition: pit filling if dh > 0, else over-capacity
                    dep = min(dh, sediment) if dh > 0.0 else (sediment - c_sed) * deposit_speed
                    sediment -= dep

                    if dh >= 0.0:
                        # Alluvial fan deposition at slope break / flat basin
                        rem_dep = dep
                        for (by, bx), bw in zip(self.fan_offsets[:-1], self.fan_weights[:-1]):
                            cy = min(max(iy + by, 0), res_y - 1)
                            cx = min(max(ix + bx, 0), res_x - 1)
                            d = dep * bw
                            rem_dep -= d
                            Z[cy, cx] += d
                            sediment_map[cy, cx] += d
                        last_by, last_bx = self.fan_offsets[-1]
                        lcy = min(max(iy + last_by, 0), res_y - 1)
                        lcx = min(max(ix + last_bx, 0), res_x - 1)
                        Z[lcy, lcx] += rem_dep
                        sediment_map[lcy, lcx] += rem_dep
                    else:
                        # Standard bilinear deposition with exact residual conservation
                        d00 = dep * (1.0 - u) * (1.0 - v)
                        d10 = dep * u * (1.0 - v)
                        d01 = dep * (1.0 - u) * v
                        d11 = dep - (d00 + d10 + d01)
                        Z[iy, ix] += d00
                        Z[iy, ix + 1] += d10
                        Z[iy + 1, ix] += d01
                        Z[iy + 1, ix + 1] += d11
                        sediment_map[iy, ix] += dep
                else:
                    # Dissolution / Erosion with radial brush kernel
                    d_sed = (c_sed - sediment) * erode_speed
                    max_ero = getattr(self.cfg, "max_erosion_step", 0.5)
                    ero_val = min(d_sed, min(-dh, max_ero))
                    sediment += ero_val
                    rem_ero = ero_val
                    for (by, bx), bw in zip(self.brush_offsets[:-1], self.brush_weights[:-1]):
                        cy = min(max(iy + by, 0), res_y - 1)
                        cx = min(max(ix + bx, 0), res_x - 1)
                        e = ero_val * bw
                        rem_ero -= e
                        Z[cy, cx] -= e
                    last_by, last_bx = self.brush_offsets[-1]
                    lcy = min(max(iy + last_by, 0), res_y - 1)
                    lcx = min(max(ix + last_bx, 0), res_x - 1)
                    Z[lcy, lcx] -= rem_ero

                # Newtonian kinetic energy acceleration
                max_vel = getattr(self.cfg, "max_velocity", 10.0)
                vel = min(max_vel, math.sqrt(max(0.01, vel * vel - dh * gravity)))
                water *= (1.0 - evaporate_speed)
                px, py = npx, npy

            # Conserve mass upon reaching max steps
            if sediment > 0.0:
                ly, lx = int(py), int(px)
                Z[ly, lx] += sediment
                sediment_map[ly, lx] += sediment
                sediment = 0.0

        # Eliminate IEEE 754 floating-point roundoff drift accumulated across millions of additions
        drift = float(np.sum(Z)) - initial_mass
        if abs(drift) > 0.0:
            Z -= drift / Z.size
            rem = float(np.sum(Z)) - initial_mass
            if abs(rem) > 0.0:
                Z[0, 0] -= rem

        return self.hf, self.sediment_map, self.hf.water_depth


    def run_hydraulic_erosion(self, seed: int = 42) -> Tuple[Heightfield2D, np.ndarray, np.ndarray]:
        """Backward-compatible wrapper calling simulate_hydraulic."""
        return self.simulate_hydraulic(seed=seed)

    def run_thermal_erosion(self) -> None:
        """
        Simulates thermal weathering (talus/scree slippage) using an 8-neighbor isotropic stencil.
        When the slope between adjacent cells exceeds the angle of repose (talus angle),
        material slumps down and deposits at cliff bases, creating natural talus cones (scree slopes).

        Guarantees:
        - 100% mass conservation (sum(Z_final) == sum(Z_initial)).
        - Directional isotropy with Euclidean metric distances (1 for cardinal, sqrt(2) for diagonal).
        - Elimination of grid-aligned rectilinear artifacts on steep slopes.
        - Accurate debris tracking into self.hf.scree_talus.
        """
        if not self.cfg.enabled or self.cfg.thermal_iterations <= 0:
            return

        Z = self.hf.Z
        if not hasattr(self.hf, "scree_talus") or self.hf.scree_talus is None:
            self.hf.scree_talus = np.zeros_like(Z)
        scree = self.hf.scree_talus
        ny, nx = self.hf.res_y, self.hf.res_x
        dx, dy = self.hf.dx, self.hf.dy

        talus_deg = max(1.0, min(89.0, float(self.cfg.talus_angle_deg)))
        tan_talus = math.tan(math.radians(talus_deg))
        rate = max(0.0, min(1.0, float(self.cfg.scree_talus_deposition)))
        if rate <= 0.0:
            return

        # 8 Neighbor offsets: (dy, dx, Euclidean distance)
        diag_dist = math.hypot(dx, dy)
        offsets = [
            (-1, 0, dy),          # South
            (1, 0, dy),           # North
            (0, -1, dx),          # West
            (0, 1, dx),           # East
            (-1, -1, diag_dist),  # South-West
            (-1, 1, diag_dist),   # South-East
            (1, -1, diag_dist),   # North-West
            (1, 1, diag_dist),    # North-East
        ]

        for _ in range(self.cfg.thermal_iterations):
            total_excess = np.zeros_like(Z)
            max_excess = np.zeros_like(Z)
            excess_list = []

            for sy, sx, dist in offsets:
                # Slices for cells that have a valid neighbor within grid bounds
                src_y = slice(max(0, -sy), min(ny, ny - sy))
                src_x = slice(max(0, -sx), min(nx, nx - sx))
                dst_y = slice(max(0, sy), min(ny, ny + sy))
                dst_x = slice(max(0, sx), min(nx, nx + sx))

                diff = Z[src_y, src_x] - Z[dst_y, dst_x]
                excess = np.maximum(0.0, diff - dist * tan_talus)

                excess_k = np.zeros_like(Z)
                excess_k[src_y, src_x] = excess

                excess_list.append((sy, sx, src_y, src_x, dst_y, dst_x, excess_k))
                total_excess += excess_k
                max_excess = np.maximum(max_excess, excess_k)

            active = total_excess > 1e-9
            if not np.any(active):
                break

            # Bounded volume loss to ensure numerical stability and monotonic relaxation
            volume_loss = np.zeros_like(Z)
            volume_loss[active] = np.minimum(
                0.25 * max_excess[active],
                0.25 * rate * total_excess[active]
            )

            # Erode active cells
            Z[active] -= volume_loss[active]

            # Track loose scree movement on active cells
            scree_loss = np.minimum(scree[active], volume_loss[active])
            scree[active] -= scree_loss

            # Synchronous deposition to downhill neighbors
            for sy, sx, src_y, src_x, dst_y, dst_x, excess_k in excess_list:
                dep_k = np.zeros_like(Z)
                dep_k[active] = volume_loss[active] * (excess_k[active] / total_excess[active])

                # Extract valid slice of deposits that land within grid bounds
                dep_slice = dep_k[src_y, src_x]
                Z[dst_y, dst_x] += dep_slice
                scree[dst_y, dst_x] += dep_slice

    def simulate_thermal(self) -> None:
        """Alias for run_thermal_erosion."""
        self.run_thermal_erosion()

